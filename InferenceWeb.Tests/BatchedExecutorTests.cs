// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

/// <summary>
/// Tests for the **batched** path through <see cref="BatchExecutor"/>: a
/// model that implements <see cref="IBatchedPagedModel"/> gets its scheduled
/// sequences packed into a single <see cref="BatchedForwardContext"/> and
/// dispatched through one <c>ForwardBatch</c> call. This is the vLLM-style
/// "one kernel many sequences" path that the legacy per-sequence swap
/// executor cannot do.
///
/// The tests use stub models so they're fast and deterministic; they verify:
///   * the scheduler packs N sequences into the batch metadata correctly,
///   * <c>ForwardBatch</c> sees the expected query starts / positions /
///     slot mappings / block tables,
///   * per-sequence logits are routed back to the right sequence,
///   * the standalone <see cref="ManagedPagedAttention"/> kernel produces
///     numerically correct output for a hand-checkable single-sequence
///     case.
/// </summary>
public class BatchedExecutorTests
{
    private const int BlockSize = 8;
    private const int VocabSize = 16;
    private const int NumLayers = 2;
    private const int NumKVHeads = 2;
    private const int HeadDim = 4;

    [Fact]
    public void ManagedPagedAttention_SingleSeqSinglePos_MatchesHandComputed()
    {
        // 1 sequence, 1 query token, 1 KV head, 4 head dim, 1 block.
        // Q = [1, 0, 0, 0]
        // Block 0 K[0] = [1, 0, 0, 0], V[0] = [10, 0, 0, 0]
        // Causal mask + scale 1/sqrt(headDim).
        // Score = Q.K = 1 * 1/2 = 0.5; softmax = 1.0; output = V[0] = [10, 0, 0, 0]

        int numHeads = 1, numKvHeads = 1, headDim = 4, blockSize = 1;
        var q = new float[] { 1, 0, 0, 0 };
        var kBlocks = new float[] { 1, 0, 0, 0 };
        var vBlocks = new float[] { 10, 0, 0, 0 };
        var output = new float[4];

        ManagedPagedAttention.Forward(
            q, kBlocks, vBlocks, output,
            numTokens: 1, numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim,
            blockSize: blockSize,
            queryStartLoc: new[] { 0, 1 },
            seqLens: new[] { 1 },
            positions: new[] { 0 },
            blockTables: new[] { new[] { 0 } },
            numSeqs: 1,
            scale: 1f / MathF.Sqrt(headDim),
            causal: true);

        Assert.Equal(10f, output[0], 4);
        Assert.Equal(0f, output[1], 4);
        Assert.Equal(0f, output[2], 4);
        Assert.Equal(0f, output[3], 4);
    }

    [Fact]
    public void ManagedPagedAttention_CausalMaskedTwoKeys_Average()
    {
        // 1 sequence, 1 query at pos=1, 2 K/V slots in block 0.
        // K[0] = K[1] = [1,0,0,0]; V[0]=[1,0,0,0], V[1]=[3,0,0,0].
        // Scores equal -> softmax 0.5/0.5; output = (1+3)/2 = [2,0,0,0].
        int numHeads = 1, numKvHeads = 1, headDim = 4, blockSize = 2;
        var q = new float[] { 1, 0, 0, 0 };
        var kBlocks = new float[] { 1, 0, 0, 0,   1, 0, 0, 0 };
        var vBlocks = new float[] { 1, 0, 0, 0,   3, 0, 0, 0 };
        var output = new float[4];

        ManagedPagedAttention.Forward(
            q, kBlocks, vBlocks, output,
            numTokens: 1, numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim,
            blockSize: blockSize,
            queryStartLoc: new[] { 0, 1 },
            seqLens: new[] { 2 },
            positions: new[] { 1 }, // attends positions 0 and 1
            blockTables: new[] { new[] { 0 } },
            numSeqs: 1,
            scale: 1f,
            causal: true);

        Assert.Equal(2f, output[0], 3);
    }

    [Fact]
    public void ManagedPagedAttention_SlidingWindow_TruncatesOldKeys()
    {
        int numHeads = 1, numKvHeads = 1, headDim = 1, blockSize = 4;
        var q = new float[] { 1 };
        var kBlocks = new float[] { 1, 1, 1, 1 };
        var vBlocks = new float[] { 1, 3, 7, 9 };
        var output = new float[1];

        ManagedPagedAttention.Forward(
            q, kBlocks, vBlocks, output,
            numTokens: 1, numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim,
            blockSize: blockSize,
            queryStartLoc: new[] { 0, 1 },
            seqLens: new[] { 4 },
            positions: new[] { 3 },
            blockTables: new[] { new[] { 0 } },
            numSeqs: 1,
            scale: 1f,
            causal: true,
            slidingWindow: 2);

        Assert.Equal(8f, output[0], 3);
    }

    [Fact]
    public void BatchExecutor_PrefersForwardBatch_WhenModelImplementsBatchedInterface()
    {
        var model = new BatchedStubModel("fp-batched", peakToken: 7);
        var cfg = SmallConfig();
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);

        var handles = new List<InferenceRequestHandle>();
        for (int i = 0; i < 4; i++)
        {
            var seq = new SequenceState($"r{i}", Enumerable.Range(1, 5 + i).ToList(),
                maxNewTokens: 4, BlockSize, SamplingConfig.Default);
            handles.Add(engine.SubmitRequest(seq));
        }

        foreach (var h in handles)
        {
            var completion = h.Completion.GetAwaiter().GetResult();
            Assert.True(completion.OutputTokenCount > 0);
        }

        // The model received N>=1 batched calls; each saw multiple sequences
        // (proving the executor really packed them, not just looped per-seq).
        Assert.True(model.NumBatchCalls > 0,
            "ForwardBatch was never called - executor probably fell back to per-sequence.");
        Assert.True(model.MaxSequencesInAnyBatch >= 2,
            $"Expected at least one batch with >=2 sequences; biggest batch had {model.MaxSequencesInAnyBatch}.");
    }

    [Fact]
    public void BatchExecutor_BatchedPath_RoutesPerSeqLogitsCorrectly()
    {
        // Each sequence carries its requested peakToken in its UserTag. The
        // stub model reads UserTag to decide which token to favour, so we
        // can verify that the per-seq logits route back to the right
        // sequence in the streamed output.
        var model = new PerSeqRoutingStubModel("fp-route");
        using var engine = new InferenceEngine(model, SmallConfig(), NullLogger.Instance);

        var handles = new List<(InferenceRequestHandle handle, int expected)>();
        var expected = new[] { 2, 5, 9, 13 };
        for (int i = 0; i < expected.Length; i++)
        {
            var seq = new SequenceState($"r{i}", Enumerable.Range(1, 4).ToList(),
                maxNewTokens: 3, BlockSize, SamplingConfig.Default, userTag: expected[i]);
            handles.Add((engine.SubmitRequest(seq), expected[i]));
        }

        foreach (var (h, expectedToken) in handles)
        {
            h.Completion.GetAwaiter().GetResult();
            Assert.Contains(expectedToken, h.Sequence.OutputTokens);
        }
    }

    [Fact]
    public async Task BatchExecutor_DecodeTimeBatchDecline_DoesNotDoubleAppendSampledToken()
    {
        var model = new DecodeDecliningBatchedStubModel("fp-decode-decline", peakToken: 7);
        using var engine = new InferenceEngine(model, SmallConfig(), NullLogger.Instance);

        var sequences = Enumerable.Range(0, 3)
            .Select(i => new SequenceState($"decline-{i}", Enumerable.Range(1, 4).ToList(),
                maxNewTokens: 5, BlockSize, SamplingConfig.Greedy))
            .ToList();
        var handles = sequences.Select(seq => engine.SubmitRequest(seq)).ToList();

        foreach (var handle in handles)
        {
            var completion = await handle.Completion.WaitAsync(TimeSpan.FromSeconds(5));
            Assert.Equal(SequenceStatus.FinishedLengthCapped, completion.Status);
            Assert.Equal(5, completion.OutputTokenCount);
            Assert.Equal(Enumerable.Repeat(7, 5), handle.Sequence.OutputTokens);
        }

        Assert.Equal(1, model.DecodeDeclines);
        Assert.Equal(0, model.NumForwardCalls);
        Assert.True(model.SingletonBatchCalls > 0,
            "The declined batch did not exercise the state-safe singleton paged fallback.");
    }

    [Fact]
    public async Task BatchExecutor_BatchedDecode_DeliversSampledTokensNotYetInTokenList()
    {
        // Regression: the executor commits a decode step's sampled token to
        // the sequence ONLY after ForwardBatch accepts the batch (so a decline
        // can fall back to per-sequence without double-appending). The batched
        // models read the batch's input tokens from OverrideFlatTokens and
        // would otherwise call seq.TokenAt(NumComputedTokens), which throws
        // ArgumentOutOfRangeException because the sampled token is not in the
        // sequence's token list yet.
        var model = new StatefulMigrationStubModel("fp-decode-override", peakToken: 7);
        using var engine = new InferenceEngine(model, SmallConfig(), NullLogger.Instance);

        var sequences = Enumerable.Range(0, 3)
            .Select(i => new SequenceState($"override-{i}", Enumerable.Range(1, 4).ToList(),
                maxNewTokens: 4, BlockSize, SamplingConfig.Greedy))
            .ToList();
        var handles = sequences.Select(seq => engine.SubmitRequest(seq)).ToList();

        foreach (var handle in handles)
        {
            var completion = await handle.Completion.WaitAsync(TimeSpan.FromSeconds(5));
            Assert.Equal(SequenceStatus.FinishedLengthCapped, completion.Status);
            Assert.Equal(4, completion.OutputTokenCount);
            Assert.Equal(Enumerable.Repeat(7, 4), handle.Sequence.OutputTokens);
        }

        Assert.True(model.DecodeRowsServedFromOverride > 0,
            "ForwardBatch never received a decode row's sampled token via OverrideFlatTokens.");
        Assert.True(model.DecodeRowsMissingOverride == 0,
            "A decode row reached ForwardBatch without the sampled token (the pre-fix crash path).");
        Assert.True(model.PagedHistoryValidationFailures == 0);
        Assert.True(model.LinearForwardCalls == 0);
    }

    [Fact]
    public void BatchExecutor_BatchDeclineAfterLinearMigration_PreservesOwnerTail()
    {
        using var model = new StatefulMigrationStubModel(
            "fp-migration-decline", peakToken: 7, declineMultiBatchNumber: 1);
        var cfg = SmallConfig();
        var pool = new BlockPool(
            cfg.NumBlocks, cfg.BlockSize, model.ComputeKVBlockByteSize(cfg.BlockSize));
        var scheduler = new ContinuousBatchScheduler(
            cfg, pool, model.KVStateFingerprint, NullLogger.Instance);
        var executor = new BatchExecutor(model, pool, scheduler, NullLogger.Instance);

        var owner = new SequenceState(
            "migration-owner",
            Enumerable.Range(1, BlockSize + 3).ToList(),
            maxNewTokens: 4,
            BlockSize,
            SamplingConfig.Greedy);
        scheduler.Submit(owner);
        var firstResults = executor.ExecuteStep(scheduler.Schedule());
        Assert.Single(firstResults);
        Assert.Null(firstResults[0].Error);

        var newcomer = new SequenceState(
            "migration-newcomer",
            Enumerable.Range(2, 4).ToList(),
            maxNewTokens: 4,
            BlockSize,
            SamplingConfig.Greedy);
        scheduler.Submit(newcomer);
        var mixedStep = scheduler.Schedule();
        Assert.Equal(2, mixedStep.ScheduledWork.Count);

        var fallbackResults = executor.ExecuteStep(mixedStep);

        Assert.All(fallbackResults, result => Assert.Null(result.Error));
        Assert.Equal(1, model.MigrationCalls);
        // The per-sequence fairness policy serves the fresh newcomer first.
        // On the next step the prior owner is restored from its captured blocks
        // and migrated again; that is where a missing partial tail is exposed.
        Assert.All(executor.ExecuteStep(scheduler.Schedule()), result => Assert.Null(result.Error));
        Assert.True(model.SawMigratedOwnerFallbackForward,
            $"The owner did not continue from the complete live linear tail after ForwardBatch declined " +
            $"(expected {model.OwnerFallbackExpectedLength}, saw {model.OwnerFallbackActualLength}).");
        Assert.False(model.LostMigratedOwnerTail,
            "The model's copy-only migration was treated as a destructive handoff before ForwardBatch accepted.");
    }

    [Fact]
    public void BatchExecutor_PagedBatchDecline_UsesPagedSingletonsAndNextBatchSeesFullHistory()
    {
        using var model = new StatefulMigrationStubModel(
            "fp-paged-decline", peakToken: 7, declineMultiBatchNumber: 2);
        var cfg = SmallConfig();
        var pool = new BlockPool(
            cfg.NumBlocks, cfg.BlockSize, model.ComputeKVBlockByteSize(cfg.BlockSize));
        var scheduler = new ContinuousBatchScheduler(
            cfg, pool, model.KVStateFingerprint, NullLogger.Instance);
        var executor = new BatchExecutor(model, pool, scheduler, NullLogger.Instance);

        var sequences = Enumerable.Range(0, 3)
            .Select(i => new SequenceState(
                $"paged-{i}",
                Enumerable.Range(1 + i, 4).ToList(),
                maxNewTokens: 5,
                BlockSize,
                SamplingConfig.Greedy))
            .ToList();
        foreach (var sequence in sequences)
            scheduler.Submit(sequence);

        Assert.All(executor.ExecuteStep(scheduler.Schedule()), result => Assert.Null(result.Error));
        Assert.All(executor.ExecuteStep(scheduler.Schedule()), result => Assert.Null(result.Error));
        Assert.All(executor.ExecuteStep(scheduler.Schedule()), result => Assert.Null(result.Error));

        Assert.Equal(sequences.Count, model.SingletonBatchCalls);
        Assert.Equal(0, model.LinearForwardCalls);
        Assert.Equal(0, model.PagedHistoryValidationFailures);
        Assert.True(model.SawAcceptedMultiBatchAfterDecline,
            "The regression did not exercise an accepted multi-sequence batch after the decline.");
    }

    [Fact]
    public void BatchExecutor_DynamicBatchDisable_DrainsPagedResidentsThroughSingletonBatches()
    {
        string previousDisable = Environment.GetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED");
        try
        {
            Environment.SetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED", "0");

            using var model = new StatefulMigrationStubModel(
                "fp-dynamic-batch-disable", peakToken: 7);
            var cfg = SmallConfig();
            var pool = new BlockPool(
                cfg.NumBlocks, cfg.BlockSize, model.ComputeKVBlockByteSize(cfg.BlockSize));
            var scheduler = new ContinuousBatchScheduler(
                cfg, pool, model.KVStateFingerprint, NullLogger.Instance);
            var executor = new BatchExecutor(model, pool, scheduler, NullLogger.Instance);

            var sequences = Enumerable.Range(0, 2)
                .Select(i => new SequenceState(
                    $"disable-paged-{i}",
                    Enumerable.Range(1 + i, 4).ToList(),
                    maxNewTokens: 4,
                    BlockSize,
                    SamplingConfig.Greedy))
                .ToList();
            foreach (var sequence in sequences)
                scheduler.Submit(sequence);

            var prefillResults = executor.ExecuteStep(scheduler.Schedule());
            Assert.All(prefillResults, result => Assert.Null(result.Error));
            Assert.Equal(1, model.ForwardBatchCalls);
            Assert.Equal(0, model.SingletonBatchCalls);

            // Options are re-read at every step. Once the histories live only
            // in model-owned paged arrays, disabling multi-sequence batching
            // must drain them through singleton paged calls rather than inject
            // unrelated pooled bytes into the linear cache.
            Environment.SetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED", "1");
            var decodeResults = executor.ExecuteStep(scheduler.Schedule());

            Assert.All(decodeResults, result => Assert.Null(result.Error));
            Assert.Equal(sequences.Count, decodeResults.Count);
            Assert.Equal(sequences.Count, model.SingletonBatchCalls);
            Assert.Equal(0, model.LinearForwardCalls);
            Assert.Equal(0, model.PagedHistoryValidationFailures);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED", previousDisable);
        }
    }

    [Fact]
    public void BatchExecutor_MixedMultimodalMigrationFailure_FallsBackBeforeAdvancingEitherSubset()
    {
        var injector = new PendingRequestInjector();
        using var model = new StatefulMigrationStubModel(
            "fp-mixed-migration-failure",
            peakToken: 7,
            migrationSucceeds: false,
            injector: injector);
        var cfg = SmallConfig();
        var pool = new BlockPool(
            cfg.NumBlocks, cfg.BlockSize, model.ComputeKVBlockByteSize(cfg.BlockSize));
        var scheduler = new ContinuousBatchScheduler(
            cfg, pool, model.KVStateFingerprint, NullLogger.Instance);
        var executor = new BatchExecutor(model, pool, scheduler, NullLogger.Instance);

        var owner = new SequenceState(
            "mixed-text-owner",
            Enumerable.Range(1, BlockSize + 1).ToList(),
            maxNewTokens: 4,
            BlockSize,
            SamplingConfig.Greedy);
        scheduler.Submit(owner);
        Assert.Null(executor.ExecuteStep(scheduler.Schedule())[0].Error);

        var multimodal = new SequenceState(
            "mixed-mm",
            Enumerable.Range(3, 4).ToList(),
            maxNewTokens: 4,
            BlockSize,
            SamplingConfig.Greedy);
        injector.AddPending(multimodal.RequestId);
        scheduler.Submit(multimodal);
        var mixedStep = scheduler.Schedule();
        Assert.Equal(2, mixedStep.ScheduledWork.Count);

        var results = executor.ExecuteStep(mixedStep);

        Assert.All(results, result => Assert.Null(result.Error));
        Assert.Equal(1, model.MigrationCalls);
        Assert.Equal(0, model.ForwardBatchCalls);
        Assert.All(executor.ExecuteStep(scheduler.Schedule()), result => Assert.Null(result.Error));
        Assert.True(model.SawMigratedOwnerFallbackForward,
            $"Expected {model.OwnerFallbackExpectedLength} live tokens, saw {model.OwnerFallbackActualLength}.");
        Assert.False(model.LostMigratedOwnerTail);
    }

    [Fact]
    public async Task BatchExecutor_FusedDecodeDecline_DoesNotDrawSeededSamplerTwice()
    {
        static SamplingConfig SeededSampling() => new()
        {
            Temperature = 1f,
            TopK = 4,
            TopP = 1f,
            MinP = 0f,
            RepetitionPenalty = 1f,
            PresencePenalty = 0f,
            FrequencyPenalty = 0f,
            Seed = 1729,
        };

        static async Task<(List<int[]> outputs, PerSeqFusedStubModel model)> RunAsync(
            bool canBatchDecode, string fingerprint)
        {
            var model = new PerSeqFusedStubModel(
                fingerprint, canBatchDecode: canBatchDecode, samplingLogits: true);
            using var engine = new InferenceEngine(model, SmallConfig(), NullLogger.Instance);
            var sequences = Enumerable.Range(0, 3)
                .Select(i => new SequenceState($"{fingerprint}-{i}",
                    Enumerable.Range(1, 4).ToList(), maxNewTokens: 16,
                    BlockSize, SeededSampling()))
                .ToList();
            var handles = sequences.Select(seq => engine.SubmitRequest(seq)).ToList();

            foreach (var handle in handles)
                await handle.Completion.WaitAsync(TimeSpan.FromSeconds(5));

            return (sequences.Select(s => s.OutputTokens.ToArray()).ToList(), model);
        }

        // Both runs use the same per-sequence fused fallback and seeded samplers.
        // The only difference is whether the model first gets (and declines) the
        // token-batched fused attempt.
        var baseline = await RunAsync(canBatchDecode: false, "seeded-baseline");
        var declined = await RunAsync(canBatchDecode: true, "seeded-decline");

        Assert.True(declined.model.BatchedFusedDecodeDeclines > 0,
            "The test did not exercise the batched fused-decode decline.");
        Assert.Equal(baseline.outputs.Count, declined.outputs.Count);
        for (int i = 0; i < baseline.outputs.Count; i++)
            Assert.Equal(baseline.outputs[i], declined.outputs[i]);
    }

    [Fact]
    public void BatchExecutor_PerSeqFused_ServesConcurrentSequencesViaForwardNotForwardBatch()
    {
        // A model that opts into the per-sequence fused path
        // (SupportsPerSequenceFusedForward=true, like Gemma 4) must have its
        // concurrent sequences served by per-sequence Forward — each with its
        // own bound per-request cache — and never fall into the op-by-op
        // ForwardBatch path. That is the parallel-decode throughput fix: it
        // keeps the GPU saturated on models whose fused single-graph decode is
        // far faster than the batched op-by-op kernel.
        //
        // The model declares SupportsLinearKVMigration like Gemma 4, so a
        // single-sequence step uses the N==1 fused Forward fast path rather than
        // ForwardBatch — hence ForwardBatch must NEVER be called regardless of
        // how the scheduler interleaves admission.
        var model = new PerSeqFusedStubModel("fp-fused");
        // Long-ish generations so multiple sequences overlap for many steps.
        using var engine = new InferenceEngine(model, SmallConfig(), NullLogger.Instance);

        var handles = new List<(InferenceRequestHandle handle, string id)>();
        for (int i = 0; i < 4; i++)
        {
            var seq = new SequenceState($"r{i}", Enumerable.Range(1, 4).ToList(),
                maxNewTokens: 20, BlockSize, SamplingConfig.Default);
            handles.Add((engine.SubmitRequest(seq), $"r{i}"));
        }

        foreach (var (h, _) in handles)
        {
            var completion = h.Completion.GetAwaiter().GetResult();
            Assert.True(completion.OutputTokenCount > 0);
        }

        // The op-by-op batched path was never used for this fused-capable model.
        Assert.Equal(0, model.NumBatchCalls);
        Assert.True(model.NumForwardCalls > 0, "Per-sequence Forward was never called.");
        // Concurrency was actually served by the fused path: at some step at
        // least two distinct per-request caches were live at once.
        Assert.True(model.MaxConcurrentBoundCaches >= 2,
            $"Expected >=2 per-request caches bound concurrently; saw {model.MaxConcurrentBoundCaches}.");
        // Every request was served through its own per-request cache binding.
        foreach (var (_, id) in handles)
            Assert.Contains(id, model.BoundRequestIds);
    }

    [Fact]
    public async Task BatchExecutor_RecurrentModelWithoutCrossSequenceReuse_SkipsPooledSnapshots()
    {
        var model = new PerSeqFusedStubModel(
            "fp-recurrent-no-pool",
            supportsCrossSequenceReuse: false,
            requiresPerBlockCapture: true);
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 64,
            MaxNumRunningSequences = 4,
            MaxPrefillChunkSize = 64,
            SoloPrefillChunkSize = 64,
            NumBlocks = 16,
            BlockSize = BlockSize,
            EnablePrefixCaching = true,
            DecodeQuantumTokens = 1,
        };
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);
        var seq = new SequenceState("long-recurrent", Enumerable.Range(1, 20).ToList(),
            maxNewTokens: 2, BlockSize, SamplingConfig.Default);

        var completion = await engine.SubmitRequest(seq).Completion.WaitAsync(TimeSpan.FromSeconds(5));

        Assert.Equal(SequenceStatus.FinishedLengthCapped, completion.Status);
        Assert.Equal(0, model.ExtractCalls);
        Assert.Equal(22, model.LastPreparedContext);
    }

    [Fact]
    public async Task InferenceEngine_ExecutorStepException_CompletesErroredRequestAndContinuesWaiting()
    {
        var model = new ThrowingBatchedStubModel("fp-step-error", badRequestId: "bad", peakToken: 7);
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 256,
            MaxNumRunningSequences = 1,
            MaxPrefillChunkSize = 64,
            NumBlocks = 4,
            BlockSize = BlockSize,
            EnablePrefixCaching = false,
            DecodeQuantumTokens = 1,
        };
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);

        var badSeq = new SequenceState("bad", Enumerable.Range(1, 4).ToList(),
            maxNewTokens: 2, BlockSize, SamplingConfig.Default);
        var goodSeq = new SequenceState("good", Enumerable.Range(1, 4).ToList(),
            maxNewTokens: 2, BlockSize, SamplingConfig.Default);

        var bad = engine.SubmitRequest(badSeq);
        var good = engine.SubmitRequest(goodSeq);

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(
            () => bad.Completion.WaitAsync(TimeSpan.FromSeconds(5)));
        Assert.Contains("bad", ex.Message);

        var goodCompletion = await good.Completion.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(SequenceStatus.FinishedLengthCapped, goodCompletion.Status);
        Assert.Equal(0, badSeq.BlockTable.NumBlocks);
        Assert.Equal(SequenceStatus.FinishedError, badSeq.Status);
        Assert.Contains("bad", model.ReleasedRequestIds);
        Assert.Contains("good", model.ReleasedRequestIds);
        Assert.Equal(cfg.NumBlocks, engine.PoolStats.freeBlocks);
    }

    [Fact]
    public async Task InferenceEngine_PromptLongerThanKvPool_FailsCleanlyInsteadOfHanging()
    {
        // Repro for the reported hang: a prompt whose KV footprint exceeds the
        // whole block pool prefills until the pool is exhausted, then — being the
        // only running sequence with nothing to preempt — can never be scheduled
        // again. Pre-fix the engine spun on empty schedules forever (GPU idle,
        // "stuck forever"). Post-fix it fails the request with a capacity error
        // and stays healthy enough to serve a subsequent in-budget request.
        //
        // maxContext defaults to 0, so the pool is NOT auto-grown; this exercises
        // the deadlock guard rather than the auto-sizing path.
        var model = new BatchedStubModel("fp-capacity", peakToken: 7);
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 8,
            MaxNumRunningSequences = 4,
            MaxPrefillChunkSize = 8,
            SoloPrefillChunkSize = 8,
            NumBlocks = 4,          // pool holds 4*8 = 32 tokens of KV
            BlockSize = BlockSize,  // 8
            EnablePrefixCaching = false,
            DecodeQuantumTokens = 1,
        };
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);
        // Pool was not auto-grown (model advertises no context length).
        Assert.Equal(cfg.NumBlocks, engine.PoolStats.totalBlocks);

        // 40-token prompt needs ceil(40/8)=5 blocks > the 4-block pool.
        var tooLong = new SequenceState("too-long", Enumerable.Range(1, 40).ToList(),
            maxNewTokens: 4, BlockSize, SamplingConfig.Default);
        var shortOk = new SequenceState("short-ok", Enumerable.Range(1, 4).ToList(),
            maxNewTokens: 3, BlockSize, SamplingConfig.Default);

        var tooLongHandle = engine.SubmitRequest(tooLong);
        var shortHandle = engine.SubmitRequest(shortOk);

        // The over-length request fails rather than hanging: if the engine were
        // still spinning, WaitAsync would throw TimeoutException here instead.
        var ex = await Assert.ThrowsAsync<InvalidOperationException>(
            () => tooLongHandle.Completion.WaitAsync(TimeSpan.FromSeconds(10)));
        Assert.Contains("capacity", ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(SequenceStatus.FinishedError, tooLong.Status);
        Assert.Equal(0, tooLong.BlockTable.NumBlocks);

        // The engine recovered: the in-budget request still completes and the
        // pool is fully reclaimed.
        var completion = await shortHandle.Completion.WaitAsync(TimeSpan.FromSeconds(10));
        Assert.True(completion.OutputTokenCount > 0);
        Assert.Equal(cfg.NumBlocks, engine.PoolStats.freeBlocks);
    }

    [Fact]
    public async Task InferenceEngine_AutoSizesKvPoolToModelContext_SoLongPromptDoesNotDeadlock()
    {
        // A model that advertises a context length gets its KV block pool sized to
        // cover that context, so an in-context prompt longer than the configured
        // default pool completes instead of deadlocking (the reported hang).
        const int modelContext = 512;
        var model = new BatchedStubModel("fp-autosize", peakToken: 7, maxContext: modelContext);
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 64,
            MaxNumRunningSequences = 4,
            MaxPrefillChunkSize = 32,
            SoloPrefillChunkSize = 64,
            NumBlocks = 4,          // default would hold only 4*8 = 32 tokens
            BlockSize = BlockSize,  // 8
            EnablePrefixCaching = false,
            DecodeQuantumTokens = 1,
        };
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);

        // Pool auto-grew to cover the model's advertised context.
        int expectedBlocks = (modelContext + BlockSize - 1) / BlockSize; // 64
        Assert.Equal(expectedBlocks, engine.PoolStats.totalBlocks);

        // A 100-token prompt overflows the configured 32-token pool but fits the
        // auto-sized 512-token pool: it must complete, not hang.
        var seq = new SequenceState("long-in-context", Enumerable.Range(1, 100).ToList(),
            maxNewTokens: 4, BlockSize, SamplingConfig.Default);
        var handle = engine.SubmitRequest(seq);
        var completion = await handle.Completion.WaitAsync(TimeSpan.FromSeconds(10));
        Assert.True(completion.OutputTokenCount > 0);
    }

    [Fact]
    public async Task InferenceEngine_AutoSizedPool_CrossesFormer65536TokenBoundary()
    {
        // Mirrors the reported PDF accounting: 62,715 prompt tokens plus a
        // 4,096-token output budget. The old 256x256 pool stopped permanently
        // at 65,536; the context-sized pool must carry the sequence to 66,811.
        const int blockSize = 256;
        const int promptTokens = 62_715;
        const int outputTokens = 4_096;
        var model = new BatchedStubModel("fp-pdf-boundary", peakToken: 7, maxContext: 262_144);
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 65_536,
            MaxNumRunningSequences = 1,
            MaxPrefillChunkSize = 65_536,
            SoloPrefillChunkSize = 65_536,
            NumBlocks = 256,
            BlockSize = blockSize,
            EnablePrefixCaching = false,
            DecodeQuantumTokens = 1,
        };
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);
        var seq = new SequenceState(
            "pdf-boundary",
            Enumerable.Repeat(1, promptTokens).ToList(),
            outputTokens,
            blockSize,
            SamplingConfig.Greedy);

        var completion = await engine.SubmitRequest(seq).Completion.WaitAsync(TimeSpan.FromSeconds(10));

        Assert.Equal(SequenceStatus.FinishedLengthCapped, completion.Status);
        Assert.Equal(outputTokens, completion.OutputTokenCount);
        Assert.Equal(promptTokens + outputTokens, seq.NumComputedTokens);
        Assert.Equal(1_024, engine.PoolStats.totalBlocks);
    }

    // ----- helpers -----

    private static SchedulerConfig SmallConfig() => new()
    {
        MaxNumBatchedTokens = 256,
        MaxNumRunningSequences = 8,
        MaxPrefillChunkSize = 64,
        NumBlocks = 16,
        BlockSize = BlockSize,
        EnablePrefixCaching = false,
        DecodeQuantumTokens = 1,
    };

    /// <summary>
    /// Minimal <see cref="IBatchedPagedModel"/>. Records how many batched calls
    /// it received and the biggest batch size, and returns deterministic
    /// logits peaked at <c>peakToken</c> for every sequence.
    /// </summary>
    private sealed class BatchedStubModel : IModelArchitecture, IBatchedPagedModel
    {
        private readonly string _fp;
        private readonly int _peak;
        private readonly int _maxContext;
        private int _cacheSeqLen;

        public int NumBatchCalls { get; private set; }
        public int MaxSequencesInAnyBatch { get; private set; }

        public BatchedStubModel(string fp, int peakToken, int maxContext = 0)
        {
            _fp = fp;
            _peak = peakToken;
            _maxContext = maxContext;
            Tokenizer = new StubTokenizer(VocabSize);
        }

        public ModelConfig Config { get; } = new ModelConfig { VocabSize = VocabSize };
        public ITokenizer Tokenizer { get; }
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => true;
        public bool SupportsKVStateSnapshot => true;
        // 0 => model advertises no context length (pool keeps its configured size);
        // a positive value drives the engine's KV-pool auto-sizing.
        public int MaxContextLength => _maxContext;
        public string KVStateFingerprint => _fp;
        public long ComputeKVBlockByteSize(int tokenCount)
            => 2L * NumLayers * NumKVHeads * tokenCount * HeadDim * sizeof(float);
        public float[] Forward(int[] tokens)
        {
            // Fallback path - shouldn't be hit when ForwardBatch is wired.
            var logits = new float[VocabSize];
            logits[_peak] = 10f;
            return logits;
        }
        public void ResetKVCache() => _cacheSeqLen = 0;
        public void TruncateKVCache(int n) => _cacheSeqLen = Math.Min(_cacheSeqLen, n);
        public bool TryExtractKVBlock(int s, int n, Span<byte> dst) => true;
        public bool TryInjectKVBlock(int s, int n, ReadOnlySpan<byte> src) { _cacheSeqLen = s + n; return true; }
        public void Dispose() { }

        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
        {
            NumBatchCalls++;
            int n = ctx.Sequences.Count;
            if (n > MaxSequencesInAnyBatch) MaxSequencesInAnyBatch = n;

            var perSeqLogits = new float[n][];
            for (int i = 0; i < n; i++)
            {
                var logits = new float[VocabSize];
                logits[_peak] = 10f;
                perSeqLogits[i] = logits;
            }
            return perSeqLogits;
        }
    }

    /// <summary>
    /// Stub whose per-sequence logits depend on the sequence's
    /// <see cref="SequenceState.UserTag"/>. Used to prove that the batched
    /// executor routes per-seq logits back to the right sequence (and not
    /// e.g. swapped or all-same).
    /// </summary>
    private sealed class PerSeqRoutingStubModel : IModelArchitecture, IBatchedPagedModel
    {
        private readonly string _fp;
        private int _cacheSeqLen;

        public PerSeqRoutingStubModel(string fp)
        {
            _fp = fp;
            Tokenizer = new StubTokenizer(VocabSize);
        }

        public ModelConfig Config { get; } = new ModelConfig { VocabSize = VocabSize };
        public ITokenizer Tokenizer { get; }
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => true;
        public bool SupportsKVStateSnapshot => true;
        public string KVStateFingerprint => _fp;
        public long ComputeKVBlockByteSize(int n) => 2L * NumLayers * NumKVHeads * n * HeadDim * sizeof(float);
        public float[] Forward(int[] tokens) => new float[VocabSize];
        public void ResetKVCache() => _cacheSeqLen = 0;
        public void TruncateKVCache(int n) => _cacheSeqLen = Math.Min(_cacheSeqLen, n);
        public bool TryExtractKVBlock(int s, int n, Span<byte> dst) => true;
        public bool TryInjectKVBlock(int s, int n, ReadOnlySpan<byte> src) { _cacheSeqLen = s + n; return true; }
        public void Dispose() { }

        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
        {
            int n = ctx.Sequences.Count;
            var result = new float[n][];
            for (int i = 0; i < n; i++)
            {
                int peak = ctx.Sequences[i].UserTag is int t ? t : 0;
                var logits = new float[VocabSize];
                logits[peak] = 10f;
                result[i] = logits;
            }
            return result;
        }
    }

    /// <summary>
    /// Stateful linear/paged cache model used by the migration-decline tests.
    /// Tokens themselves stand in for K/V rows, making a missing partial tail
    /// observable without a real attention kernel.
    /// </summary>
    private sealed class StatefulMigrationStubModel : IModelArchitecture, IBatchedPagedModel
    {
        private readonly string _fp;
        private readonly int _peak;
        private readonly int _declineMultiBatchNumber;
        private readonly bool _migrationSucceeds;
        private readonly IMultimodalInjector _injector;
        private readonly List<int> _linearHistory = new();
        private readonly Dictionary<string, List<int>> _pagedHistory =
            new(StringComparer.Ordinal);
        private int _multiBatchCalls;
        private bool _declinedMultiBatch;
        private bool _awaitingOwnerFallback;
        private string _expectedOwnerRequestId;
        private int[] _expectedOwnerFallbackHistory = Array.Empty<int>();

        public StatefulMigrationStubModel(
            string fp,
            int peakToken,
            int declineMultiBatchNumber = 0,
            bool migrationSucceeds = true,
            IMultimodalInjector injector = null)
        {
            _fp = fp;
            _peak = peakToken;
            _declineMultiBatchNumber = declineMultiBatchNumber;
            _migrationSucceeds = migrationSucceeds;
            _injector = injector;
            Tokenizer = new StubTokenizer(VocabSize);
        }

        public int MigrationCalls { get; private set; }
        public int ForwardBatchCalls { get; private set; }
        public int SingletonBatchCalls { get; private set; }
        public int LinearForwardCalls { get; private set; }
        public int PagedHistoryValidationFailures { get; private set; }
        public int DecodeRowsServedFromOverride { get; private set; }
        public int DecodeRowsMissingOverride { get; private set; }
        public bool SawMigratedOwnerFallbackForward { get; private set; }
        public bool LostMigratedOwnerTail { get; private set; }
        public bool SawAcceptedMultiBatchAfterDecline { get; private set; }
        public int OwnerFallbackExpectedLength { get; private set; }
        public int OwnerFallbackActualLength { get; private set; }

        public ModelConfig Config { get; } = new ModelConfig { VocabSize = VocabSize };
        public ITokenizer Tokenizer { get; }
        public IMultimodalInjector MultimodalInjector => _injector;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => true;
        public bool SupportsKVStateSnapshot => true;
        public bool SupportsCrossSequenceKvReuse => true;
        public bool SupportsLinearKVMigration => true;
        public string KVStateFingerprint => _fp;
        public long ComputeKVBlockByteSize(int tokenCount) => tokenCount * sizeof(int);

        public float[] Forward(int[] tokens)
        {
            LinearForwardCalls++;
            if (_awaitingOwnerFallback
                && tokens.Length == 1
                && tokens[0] == _peak)
            {
                ValidateRestoredOwnerTail();
            }

            _linearHistory.AddRange(tokens);
            return PeakedLogits();
        }

        public void ResetKVCache() => _linearHistory.Clear();

        public void TruncateKVCache(int tokenCount)
        {
            if (_linearHistory.Count > tokenCount)
                _linearHistory.RemoveRange(tokenCount, _linearHistory.Count - tokenCount);
        }

        public bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            if (startToken < 0 || tokenCount < 0
                || startToken + tokenCount > _linearHistory.Count
                || destination.Length < tokenCount * sizeof(int))
            {
                return false;
            }

            for (int i = 0; i < tokenCount; i++)
            {
                BitConverter.TryWriteBytes(
                    destination.Slice(i * sizeof(int), sizeof(int)),
                    _linearHistory[startToken + i]);
            }
            return true;
        }

        public bool TryInjectKVBlock(int startToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            if (startToken < 0 || tokenCount < 0
                || startToken > _linearHistory.Count
                || source.Length < tokenCount * sizeof(int))
            {
                return false;
            }

            if (_linearHistory.Count > startToken)
                _linearHistory.RemoveRange(startToken, _linearHistory.Count - startToken);
            for (int i = 0; i < tokenCount; i++)
            {
                _linearHistory.Add(BitConverter.ToInt32(
                    source.Slice(i * sizeof(int), sizeof(int))));
            }
            return true;
        }

        public bool TryMigrateLinearKVToPaged(SequenceState owner, int blockSize)
        {
            MigrationCalls++;
            if (_awaitingOwnerFallback
                && string.Equals(owner.RequestId, _expectedOwnerRequestId, StringComparison.Ordinal))
            {
                ValidateRestoredOwnerTail();
            }

            int count = owner.NumComputedTokens;
            _expectedOwnerFallbackHistory = _linearHistory.Take(count).ToArray();
            _expectedOwnerRequestId = owner.RequestId;
            _awaitingOwnerFallback = true;
            if (!_migrationSucceeds || _linearHistory.Count < count)
                return false;

            _pagedHistory[owner.RequestId] = _expectedOwnerFallbackHistory.ToList();
            return true;
        }

        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
        {
            ForwardBatchCalls++;
            bool isMulti = ctx.Sequences.Count > 1;
            if (!isMulti)
                SingletonBatchCalls++;
            else
                _multiBatchCalls++;

            int numTokens = 0;
            for (int i = 0; i < ctx.NumScheduledTokens.Count; i++)
                numTokens += ctx.NumScheduledTokens[i];

            ValidatePagedPrefixes(ctx);

            if (isMulti
                && _declineMultiBatchNumber > 0
                && _multiBatchCalls == _declineMultiBatchNumber)
            {
                _declinedMultiBatch = true;
                throw new NotSupportedException("decline the configured multi-sequence batch");
            }

            _awaitingOwnerFallback = false;
            for (int s = 0; s < ctx.Sequences.Count; s++)
            {
                var seq = ctx.Sequences[s];
                if (!_pagedHistory.TryGetValue(seq.RequestId, out var history))
                {
                    history = new List<int>();
                    _pagedHistory[seq.RequestId] = history;
                }

                int queryStart = ctx.QueryStartLoc[s];
                int queryEnd = ctx.QueryStartLoc[s + 1];
                for (int q = queryStart; q < queryEnd; q++)
                {
                    int position = ctx.Positions[q];
                    if (history.Count != position)
                    {
                        PagedHistoryValidationFailures++;
                        if (history.Count > position)
                            history.RemoveRange(position, history.Count - position);
                        while (history.Count < position)
                            history.Add(int.MinValue);
                    }

                    // Production batched models read the batch's input tokens
                    // from OverrideFlatTokens: decode steps forward the
                    // sampled-but-not-yet-committed token, which does not
                    // exist in the sequence's token list yet (seq.TokenAt
                    // would throw ArgumentOutOfRangeException). Tally how
                    // decode rows are served so tests can assert the contract.
                    if (ctx.OverrideFlatTokens != null)
                    {
                        if (ctx.OverrideFlatTokens.Length != numTokens)
                            PagedHistoryValidationFailures++;
                        if (position >= seq.NumTotalTokens)
                            DecodeRowsServedFromOverride++;
                        history.Add(ctx.OverrideFlatTokens[q]);
                    }
                    else
                    {
                        if (position >= seq.NumTotalTokens)
                            DecodeRowsMissingOverride++;
                        history.Add(position < seq.NumTotalTokens
                            ? seq.TokenAt(position)
                            : _peak);
                    }
                }
            }

            if (isMulti && _declinedMultiBatch)
                SawAcceptedMultiBatchAfterDecline = true;

            return Enumerable.Range(0, ctx.Sequences.Count)
                .Select(_ => PeakedLogits())
                .ToArray();
        }

        private void ValidatePagedPrefixes(BatchedForwardContext ctx)
        {
            foreach (var seq in ctx.Sequences)
            {
                int actual = _pagedHistory.TryGetValue(seq.RequestId, out var history)
                    ? history.Count
                    : 0;
                if (actual != seq.NumComputedTokens)
                    PagedHistoryValidationFailures++;
            }
        }

        private float[] PeakedLogits()
        {
            var logits = new float[VocabSize];
            logits[_peak] = 10f;
            return logits;
        }

        private void ValidateRestoredOwnerTail()
        {
            OwnerFallbackExpectedLength = _expectedOwnerFallbackHistory.Length;
            OwnerFallbackActualLength = _linearHistory.Count;
            if (_linearHistory.SequenceEqual(_expectedOwnerFallbackHistory))
                SawMigratedOwnerFallbackForward = true;
            else
                LostMigratedOwnerTail = true;
            _awaitingOwnerFallback = false;
        }

        public void Dispose() { }
    }

    private sealed class PendingRequestInjector : IMultimodalInjector
    {
        private readonly HashSet<string> _pending = new(StringComparer.Ordinal);

        public void AddPending(string requestId) => _pending.Add(requestId);
        public void LoadProjectors(string mmProjPath) { }
        public List<int> ProcessPromptTokens(
            List<ChatMessage> history, List<int> inputTokens, string requestId = null)
            => inputTokens;
        public bool QueuePromptEmbeddings(int reusablePrefixTokenCount, string requestId = null)
            => requestId != null && _pending.Remove(requestId);
        public bool QueuePromptEmbeddingsForSlice(
            int promptStartToken, int tokenCount, string requestId = null)
            => requestId != null && _pending.Remove(requestId);
        public int ClampReusablePrefix(int reusablePrefixTokenCount, string requestId = null)
            => reusablePrefixTokenCount;
        public int ClampTrimStart(int trimStartTokenCount, string requestId = null)
            => trimStartTokenCount;
        public void TrimPreparedPrompt(int trimStartTokenCount, string requestId = null) { }
        public bool HasPendingEmbeddings(string requestId)
            => requestId != null && _pending.Contains(requestId);
        public void ClearPreparedPromptState(string requestId)
            => _pending.Remove(requestId);
    }

    private sealed class DecodeDecliningBatchedStubModel : IModelArchitecture, IBatchedPagedModel
    {
        private readonly string _fp;
        private readonly int _peak;
        private bool _declined;

        public DecodeDecliningBatchedStubModel(string fp, int peakToken)
        {
            _fp = fp;
            _peak = peakToken;
            Tokenizer = new StubTokenizer(VocabSize);
        }

        public int DecodeDeclines { get; private set; }
        public int NumForwardCalls { get; private set; }
        public int SingletonBatchCalls { get; private set; }
        public ModelConfig Config { get; } = new ModelConfig { VocabSize = VocabSize };
        public ITokenizer Tokenizer { get; }
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => true;
        public bool SupportsKVStateSnapshot => true;
        public string KVStateFingerprint => _fp;
        public long ComputeKVBlockByteSize(int n)
            => 2L * NumLayers * NumKVHeads * n * HeadDim * sizeof(float);

        public float[] Forward(int[] tokens)
        {
            NumForwardCalls++;
            return PeakedLogits();
        }

        public void ResetKVCache() { }
        public void TruncateKVCache(int n) { }
        public bool TryExtractKVBlock(int s, int n, Span<byte> dst) => true;
        public bool TryInjectKVBlock(int s, int n, ReadOnlySpan<byte> src) => true;
        public void Dispose() { }

        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
        {
            if (ctx.Sequences.Count == 1)
                SingletonBatchCalls++;
            bool isDecode = ctx.Sequences.Any(seq => seq.NumComputedTokens >= seq.PromptTokens.Count);
            if (isDecode && !_declined)
            {
                _declined = true;
                DecodeDeclines++;
                throw new NotSupportedException("decline one decode batch");
            }

            return Enumerable.Range(0, ctx.Sequences.Count)
                .Select(_ => PeakedLogits())
                .ToArray();
        }

        private float[] PeakedLogits()
        {
            var logits = new float[VocabSize];
            logits[_peak] = 10f;
            return logits;
        }
    }

    /// <summary>
    /// Stub that opts into the per-sequence fused path. Tracks per-request
    /// "caches" (just a bound-id set here) and returns logits for the
    /// currently-bound request, peaked at the token registered in
    /// <see cref="PeakForRequest"/>. Lets us assert the executor (a) never calls
    /// ForwardBatch, (b) binds a distinct cache per request, and (c) routes each
    /// sequence's logits correctly through per-sequence Forward.
    /// </summary>
    private sealed class PerSeqFusedStubModel : IModelArchitecture, IBatchedPagedModel
    {
        private readonly string _fp;
        private readonly bool _supportsCrossSequenceReuse;
        private readonly bool _requiresPerBlockCapture;
        private readonly bool _canBatchDecode;
        private readonly bool _samplingLogits;
        private string _activeReqId;
        private readonly HashSet<string> _liveCaches = new(StringComparer.Ordinal);

        public PerSeqFusedStubModel(
            string fp,
            bool supportsCrossSequenceReuse = true,
            bool requiresPerBlockCapture = false,
            bool canBatchDecode = true,
            bool samplingLogits = false)
        {
            _fp = fp;
            _supportsCrossSequenceReuse = supportsCrossSequenceReuse;
            _requiresPerBlockCapture = requiresPerBlockCapture;
            _canBatchDecode = canBatchDecode;
            _samplingLogits = samplingLogits;
            Tokenizer = new StubTokenizer(VocabSize);
        }

        public Dictionary<string, int> PeakForRequest { get; } = new(StringComparer.Ordinal);
        public int NumBatchCalls { get; private set; }
        public int NumForwardCalls { get; private set; }
        public int ExtractCalls { get; private set; }
        public int LastPreparedContext { get; private set; }
        public int MaxConcurrentBoundCaches { get; private set; }
        public int BatchedFusedDecodeDeclines { get; private set; }
        public HashSet<string> BoundRequestIds { get; } = new(StringComparer.Ordinal);

        public ModelConfig Config { get; } = new ModelConfig { VocabSize = VocabSize };
        public ITokenizer Tokenizer { get; }
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => true;
        public bool SupportsKVStateSnapshot => true;
        public bool SupportsCrossSequenceKvReuse => _supportsCrossSequenceReuse;
        public bool RequiresPerBlockCapture => _requiresPerBlockCapture;
        public string KVStateFingerprint => _fp;
        public long ComputeKVBlockByteSize(int n) => 2L * NumLayers * NumKVHeads * n * HeadDim * sizeof(float);

        public float[] Forward(int[] tokens)
        {
            NumForwardCalls++;
            var logits = new float[VocabSize];
            if (_samplingLogits)
            {
                Array.Fill(logits, -100f);
                for (int i = 0; i < 4; i++) logits[i] = 0f;
                return logits;
            }
            int peak = _activeReqId != null && PeakForRequest.TryGetValue(_activeReqId, out var p) ? p : 0;
            logits[peak] = 10f;
            return logits;
        }

        public void ResetKVCache() { }
        public void TruncateKVCache(int n) { }
        public void PrepareForPrefill(int totalPromptTokens) => LastPreparedContext = totalPromptTokens;
        public bool TryExtractKVBlock(int s, int n, Span<byte> dst) { ExtractCalls++; return true; }
        public bool TryInjectKVBlock(int s, int n, ReadOnlySpan<byte> src) => true;
        public void Dispose() { }

        // The executor must NOT call this for a fused-capable model at N>=2.
        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
        {
            NumBatchCalls++;
            var r = new float[ctx.Sequences.Count][];
            for (int i = 0; i < r.Length; i++) r[i] = new float[VocabSize];
            return r;
        }

        public bool SupportsPerSequenceFusedForward => true;

        // Mirror Gemma 4: linear KV migration is supported, so the executor's
        // N==1 fast path uses fused Forward (not ForwardBatch) for single steps.
        public bool SupportsLinearKVMigration => true;
        public bool TryMigrateLinearKVToPaged(SequenceState owner, int blockSize) => true;

        public bool CanBatchDecode(string requestId, int position) => _canBatchDecode;

        public bool TryForwardBatchedFusedDecode(
            IReadOnlyList<string> requestIds, int[] tokens, int[] positions, float[][] outLogits)
        {
            BatchedFusedDecodeDeclines++;
            return false;
        }

        public bool BindSequenceCache(string requestId)
        {
            BoundRequestIds.Add(requestId);
            bool fresh = _liveCaches.Add(requestId);
            _activeReqId = requestId;
            if (_liveCaches.Count > MaxConcurrentBoundCaches)
                MaxConcurrentBoundCaches = _liveCaches.Count;
            return fresh;
        }

        public void AdoptPrimaryCacheToFused(string requestId)
        {
            _liveCaches.Add(requestId);
            _activeReqId = requestId;
        }

        public void RestorePrimaryCache() => _activeReqId = null;

        public bool HasFusedSequenceCache(string requestId) => _liveCaches.Contains(requestId);

        public void OnSequenceReleased(string requestId)
        {
            _liveCaches.Remove(requestId);
            if (string.Equals(_activeReqId, requestId, StringComparison.Ordinal))
                _activeReqId = null;
        }
    }

    private sealed class ThrowingBatchedStubModel : IModelArchitecture, IBatchedPagedModel
    {
        private readonly string _fp;
        private readonly string _badRequestId;
        private readonly int _peak;

        public ThrowingBatchedStubModel(string fp, string badRequestId, int peakToken)
        {
            _fp = fp;
            _badRequestId = badRequestId;
            _peak = peakToken;
            Tokenizer = new StubTokenizer(VocabSize);
        }

        public List<string> ReleasedRequestIds { get; } = new();

        public ModelConfig Config { get; } = new ModelConfig { VocabSize = VocabSize };
        public ITokenizer Tokenizer { get; }
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => true;
        public bool SupportsKVStateSnapshot => true;
        public string KVStateFingerprint => _fp;
        public long ComputeKVBlockByteSize(int n) => 2L * NumLayers * NumKVHeads * n * HeadDim * sizeof(float);
        public float[] Forward(int[] tokens) => new float[VocabSize];
        public void ResetKVCache() { }
        public void TruncateKVCache(int n) { }
        public bool TryExtractKVBlock(int s, int n, Span<byte> dst) => true;
        public bool TryInjectKVBlock(int s, int n, ReadOnlySpan<byte> src) => true;
        public void Dispose() { }

        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
        {
            for (int i = 0; i < ctx.Sequences.Count; i++)
            {
                if (ctx.Sequences[i].RequestId == _badRequestId)
                    throw new InvalidOperationException($"boom for {_badRequestId}");
            }

            var result = new float[ctx.Sequences.Count][];
            for (int i = 0; i < result.Length; i++)
            {
                var logits = new float[VocabSize];
                logits[_peak] = 10f;
                result[i] = logits;
            }
            return result;
        }

        public void OnSequenceReleased(string requestId)
        {
            ReleasedRequestIds.Add(requestId);
        }
    }

    private sealed class StubTokenizer : ITokenizer
    {
        public StubTokenizer(int vocab)
        {
            Vocab = new string[vocab];
            for (int i = 0; i < vocab; i++) Vocab[i] = i.ToString();
        }
        public string[] Vocab { get; }
        public int BosTokenId => -1;
        public int[] EosTokenIds => Array.Empty<int>();
        public int VocabSize => Vocab.Length;
        public List<int> Encode(string text, bool addSpecial = true) => new();
        public string Decode(List<int> ids) => string.Join(",", ids);
        public void AppendTokenBytes(int tokenId, List<byte> buffer)
        {
            foreach (var b in System.Text.Encoding.UTF8.GetBytes(tokenId.ToString()))
                buffer.Add(b);
        }
        public bool IsEos(int tokenId) => false;
        public int LookupToken(string tokenStr) => -1;
    }
}
