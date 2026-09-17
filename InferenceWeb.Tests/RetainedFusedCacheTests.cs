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
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// Regression tests for cross-request KV-cache prefix reuse on the per-sequence
/// FUSED concurrent-decode path (the high-throughput path a sliding-window model
/// like Gemma 4 takes for N&gt;=2 concurrent requests).
///
/// Bug: that path keeps each request's full K/V in its own per-request holder and
/// never writes the shared paged blocks, so a finished concurrent request left
/// NOTHING in the prefix-cache pool — and a sliding-window model's pool can't
/// restore a long prefix anyway. A multi-turn follow-up ("请继续") submitted after
/// a concurrent round therefore re-prefilled the whole conversation from scratch
/// (KV-cache reuse ratio 0). The fix retains a small LRU of finished fused holders
/// and re-adopts one for a follow-up whose prompt exactly extends it.
///
/// Uses a deterministic <see cref="FusedStubModel"/> that mimics a sliding-window
/// model on the fused path (per-request holders, capped pooled reuse) without a
/// real LLM, so the scheduler/executor/engine glue is exercised end-to-end.
/// </summary>
public class RetainedFusedCacheTests
{
    private const int BlockSize = 8;
    private const int VocabSize = 16;
    private const int Cap = 16;         // sliding-window cap (pooled reuse ceiling)
    private const int PeakToken = 3;    // greedy argmax always lands here

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void AdoptionMetadataOutOfMemory_KeepsRetainedOwnerAndPoolUntilRetry(bool rewind, bool afterBlockReservation)
    {
        WithRetentionSettings(() =>
        {
            var model = new FusedStubModel { SupportsExactFusedCacheReuse = true, SupportsPrefixCheckpoints = false };
            var executor = NewRetentionExecutor(model);
            PrimeFinishedHolder(executor, model, "source", 1, 128);
            Assert.True(executor.TryRetainReleasedFusedCache("source"));
            int reused = rewind ? 104 : 128;
            var prompt = Enumerable.Repeat(1, reused).Concat(Enumerable.Repeat(2, 8)).ToList();
            var next = new SequenceState("next", prompt, 4, BlockSize, SamplingConfig.Greedy);
            var reserve = executor.ReserveRetainedAdoptionMetadata;
            var pool = ExecutorPool(executor);
            int free = pool.NumFreeBlocks;
            executor.ReserveRetainedAdoptionMetadata = (seq, count) =>
            {
                Assert.True(count > 8, "fixture must require block-table growth");
                if (afterBlockReservation) seq.BlockTable.EnsureBlockCapacity(count);
                throw new OutOfMemoryException("controlled adoption metadata allocation");
            };

            Assert.False(executor.TryAdoptFusedContinuation(next, reused));
            Assert.Equal(0, model.RebindCalls);
            Assert.True(model.HasRetainedHolder("source"));
            Assert.False(model.HasFusedSequenceCache("next"));
            Assert.Single(RetainedMetadata(executor));
            Assert.Equal(free, pool.NumFreeBlocks);
            Assert.Equal(0, next.BlockTable.NumBlocks);
            Assert.Equal(0, next.NumComputedTokens);
            Assert.Equal(0, next.PrefixCacheReusedTokens);
            Assert.Empty(PendingRetainedTruncations(executor));

            executor.ReserveRetainedAdoptionMetadata = reserve;
            Assert.True(executor.TryAdoptFusedContinuation(next, reused));
            Assert.Equal(1, model.RebindCalls);
            Assert.False(model.HasRetainedHolder("source"));
            Assert.True(model.HasFusedSequenceCache("next"));
            Assert.Empty(RetainedMetadata(executor));
            Assert.Equal(reused, next.NumComputedTokens);
            Assert.Equal(reused, next.PrefixCacheReusedTokens);
            Assert.Equal(free - reused / BlockSize, pool.NumFreeBlocks);
            if (rewind) Assert.Equal(reused, PendingRetainedTruncations(executor)["next"]);
            else Assert.Empty(PendingRetainedTruncations(executor));
            executor.DiscardReleasedFusedCacheBookkeeping("next");
            model.OnSequenceReleased("next");
            pool.Free(next.BlockTable.Clear());
            Assert.Equal(free, pool.NumFreeBlocks);
            Assert.False(model.HasFusedSequenceCache("next"));
            Assert.Empty(PendingRetainedTruncations(executor));
        });
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RetentionMetadataOutOfMemory_DeclinesBeforeChangingNativeOwnership(bool reusedId)
    {
        WithRetentionSettings(() =>
        {
            var model = new FusedStubModel();
            var executor = NewRetentionExecutor(model);
            const string key = "allocation-owner";
            if (reusedId)
            {
                PrimeFinishedHolder(executor, model, key, 1);
                Assert.True(executor.TryRetainReleasedFusedCache(key));
            }
            PrimeFinishedHolder(executor, model, key, 5);
            int transfers = model.RetainCalls;
            executor.AllocateRetainedCacheTokens = _ => throw new OutOfMemoryException("controlled metadata allocation");

            Assert.False(executor.TryRetainReleasedFusedCache(key));
            Assert.Equal(transfers, model.RetainCalls);
            Assert.True(model.HasFusedSequenceCache(key));
            Assert.Equal(reusedId, model.HasRetainedHolder(key));
            Assert.Equal(reusedId ? 1 : 0, RetainedMetadata(executor).Length);
            Assert.Empty(model.DiscardedRetainedRequestIds);
            if (reusedId) Assert.All(RetainedMetadata(executor)[0].Tokens, token => Assert.Equal(1, token));

            // The normal release hook still owns the active request, while a
            // previous same-id retained cache remains independently tracked.
            model.OnSequenceReleased(key);
            Assert.False(model.HasFusedSequenceCache(key));
            Assert.Equal(reusedId, model.HasRetainedHolder(key));
            executor.Reset();
            Assert.False(model.HasRetainedHolder(key));
        });
    }

    [Theory]
    [InlineData("same-id")]
    [InlineData("budget")]
    [InlineData("trim")]
    public void RetentionDiscardFailure_PreservesMetadataAndOwnerUntilSuccessfulRetry(string operation)
    {
        WithRetentionSettings(() =>
        {
            var model = new FusedStubModel();
            var executor = NewRetentionExecutor(model);
            PrimeFinishedHolder(executor, model, "old", 1);
            Assert.True(executor.TryRetainReleasedFusedCache("old"));
            string nextKey = operation == "same-id" ? "old" : "new";
            var next = PrimeFinishedHolder(executor, model, nextKey, 5);
            if (operation == "trim") Assert.True(executor.TryRetainReleasedFusedCache(nextKey));
            if (operation == "budget") Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE_MAX", "1");
            model.ThrowOnDiscardKey = "old";

            if (operation == "trim") Assert.Throws<InvalidOperationException>(() => executor.TrimIdleMemory());
            else Assert.Throws<InvalidOperationException>(() => executor.TryRetainReleasedFusedCache(nextKey));
            Assert.True(model.HasRetainedHolder("old"));
            Assert.Equal(operation == "same-id" ? 1 : 2, RetainedMetadata(executor).Length);
            Assert.All(RetainedMetadata(executor)[0].Tokens, token => Assert.Equal(1, token));
            Assert.Empty(model.DiscardedRetainedRequestIds);
            Assert.Equal(operation == "same-id" ? 1 : 2, model.RetainCalls);
            Assert.Equal(operation == "same-id", model.HasFusedSequenceCache(nextKey));

            model.ThrowOnDiscardKey = null;
            if (operation == "same-id")
            {
                TrackFinishedHolder(executor, next);
                Assert.True(executor.TryRetainReleasedFusedCache(nextKey));
            }
            else Assert.Contains("evicted 1 retained holder", executor.TrimIdleMemory());
            Assert.Equal(new[] { "old" }, model.DiscardedRetainedRequestIds);
            Assert.Single(RetainedMetadata(executor));
            Assert.All(RetainedMetadata(executor)[0].Tokens, token => Assert.Equal(5, token));
            Assert.True(model.HasRetainedHolder(nextKey));
            executor.Reset();
            Assert.Empty(RetainedMetadata(executor));
            Assert.False(model.HasRetainedHolder(nextKey));
        });
    }

    private static void WithRetentionSettings(Action body)
    {
        string enabled = Environment.GetEnvironmentVariable("TS_RETAINED_FUSED_CACHE");
        string budget = Environment.GetEnvironmentVariable("TS_RETAINED_FUSED_CACHE_MAX");
        try
        {
            Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE", "1");
            Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE_MAX", "4");
            body();
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE", enabled);
            Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE_MAX", budget);
        }
    }

    private static BatchExecutor NewRetentionExecutor(FusedStubModel model)
    {
        var cfg = Config();
        var pool = new BlockPool(cfg.NumBlocks, cfg.BlockSize, model.ComputeKVBlockByteSize(cfg.BlockSize));
        var scheduler = new ContinuousBatchScheduler(cfg, pool, model.KVStateFingerprint, NullLogger.Instance);
        return new BatchExecutor(model, pool, scheduler, NullLogger.Instance);
    }

    private static SequenceState PrimeFinishedHolder(BatchExecutor executor, FusedStubModel model, string key, int token, int length = 32)
    {
        var seq = new SequenceState(key, Enumerable.Repeat(token, length).ToList(), 1, BlockSize, SamplingConfig.Greedy);
        var pool = ExecutorPool(executor);
        var blocks = pool.AllocateNew((length + BlockSize - 1) / BlockSize) ?? throw new InvalidOperationException("test block pool exhausted");
        foreach (var block in blocks) seq.BlockTable.AppendBlock(block);
        Assert.True(model.BindSequenceCache(key));
        model.Forward(seq.PromptTokens.ToArray());
        model.RestorePrimaryCache();
        seq.AdvanceComputedTokens(length);
        seq.Status = SequenceStatus.FinishedLengthCapped;
        TrackFinishedHolder(executor, seq);
        return seq;
    }

    private static BlockPool ExecutorPool(BatchExecutor executor)
    {
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        return (BlockPool)typeof(BatchExecutor).GetField("_pool", flags)!.GetValue(executor)!;
    }

    private static Dictionary<string, int> PendingRetainedTruncations(BatchExecutor executor)
    {
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        return (Dictionary<string, int>)typeof(BatchExecutor).GetField("_pendingRetainedFusedTruncations", flags)!.GetValue(executor)!;
    }

    private static void TrackFinishedHolder(BatchExecutor executor, SequenceState seq)
    {
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        var tracked = (Dictionary<string, SequenceState>)typeof(BatchExecutor).GetField("_fusedSeqById", flags)!.GetValue(executor)!;
        tracked[seq.RequestId] = seq;
    }

    private static (string Key, int[] Tokens)[] RetainedMetadata(BatchExecutor executor)
    {
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        var entries = (System.Collections.IEnumerable)typeof(BatchExecutor).GetField("_retainedFused", flags)!.GetValue(executor)!;
        return entries.Cast<object>().Select(entry => (
            (string)entry.GetType().GetField("RequestId")!.GetValue(entry)!,
            (int[])entry.GetType().GetField("Tokens")!.GetValue(entry)!)).ToArray();
    }

    [Theory]
    [InlineData(false, false, false)]
    [InlineData(true, false, false)]
    [InlineData(true, true, false)]
    [InlineData(true, false, true)]
    public async Task SlotAwareRetainedRewind_AlignsLongSuffixOrDeclinesWithoutInflatedReuse(
        bool enabled, bool rejectQuery, bool refuseExecution)
    {
        await WithCheckpointsOnAsync(async () =>
        {
            var model = new FusedStubModel(supportsCrossSequenceKvReuse: false,
                refuseTruncationAtExecution: refuseExecution)
            {
                SupportsPrefixCheckpoints = false, SupportsExactFusedCacheReuse = enabled,
                ExactQueryHeadOffset = rejectQuery ? 1 : 0, KVCacheTruncationGranularity = 2,
            };
            var gate = new ComputeGate();
            using var engine = new InferenceEngine(model, Config(), NullLogger.Instance) { ComputeGate = gate };
            async Task<InferenceCompletion[]> Round(string name, bool follow)
            {
                SequenceState Request(int branch)
                {
                    var prompt = Enumerable.Repeat(branch, 65).ToList();
                    if (follow) prompt.AddRange(Enumerable.Repeat(PeakToken + 1, 4));
                    return new SequenceState(name + branch, prompt, follow ? 4 : 32, BlockSize, SamplingConfig.Greedy);
                }
                gate.Close();
                long held = engine.StepsHeldByGate;
                var a = engine.SubmitRequest(Request(1));
                using (var timeout = new System.Threading.CancellationTokenSource(TimeSpan.FromSeconds(5)))
                    while (engine.StepsHeldByGate == held) await Task.Delay(1, timeout.Token);
                var b = engine.SubmitRequest(Request(2));
                gate.Open();
                var da = DrainAsync(a); var db = DrainAsync(b);
                await Task.WhenAll(da, db);
                return new[] { (await da).completion, (await db).completion };
            }
            await Round("slot-first-", false);
            Assert.True(model.HasRetainedHolder("slot-first-1"));
            Assert.True(model.HasRetainedHolder("slot-first-2"));
            var completed = await Round("slot-next-", true);
            int expected = enabled && !rejectQuery && !refuseExecution ? 64 : 0;
            foreach (var item in completed)
            {
                Assert.Equal(SequenceStatus.FinishedLengthCapped, item.Status);
                Assert.Equal(expected, item.PrefixCacheReusedTokens);
                Assert.Equal(69, item.PromptTokenCount);
            }
            if (enabled)
                Assert.Contains(model.ExactQueries, q => q.Retained && q.Target == 64 && q.Cached - q.Target > 16);
            else Assert.Empty(model.ExactQueries);
            if (expected > 0) Assert.All(model.TruncationTargets, target => Assert.Equal(64, target));
            else Assert.Empty(model.TruncationTargets);
            Assert.Equal(enabled && !rejectQuery && refuseExecution ? 2 : 0, model.ExecutionTruncationRefusals);
            foreach (string request in new[] { "slot-next-1", "slot-next-2" })
                Assert.Contains(model.ForwardCalls, call => call.RequestId == request && call.Start == expected);
        });
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task SlotAwareLiveRewind_UsesActualHolderBeyondTheDefaultLimit(bool enabled)
    {
        await WithCheckpointsOnAsync(async () =>
        {
            var model = new FusedStubModel(supportsCrossSequenceKvReuse: false)
            { SupportsPrefixCheckpoints = false, SupportsExactFusedCacheReuse = enabled,
              KVCacheTruncationGranularity = 2 };
            using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);
            await DrainAsync(engine.SubmitRequest(new SequenceState("live-first",
                Enumerable.Repeat(1, 65).ToList(), 32, BlockSize, SamplingConfig.Greedy)));
            var prompt = Enumerable.Repeat(1, 65).Concat(Enumerable.Repeat(PeakToken + 1, 4)).ToList();
            var follow = await DrainAsync(engine.SubmitRequest(new SequenceState("live-next",
                prompt, 4, BlockSize, SamplingConfig.Greedy)));
            Assert.Equal(enabled ? 64 : 0, follow.completion.PrefixCacheReusedTokens);
            if (enabled) Assert.Contains(model.ExactQueries, q => !q.Retained && q.Target == 64 && q.Cached - q.Target > 16);
            else Assert.Empty(model.ExactQueries);
        });
    }

    [Fact]
    public async Task ConcurrentRound_ThenParallelFollowUps_ReuseFullPrefix()
    {
        // ---- Reproduce the bug: retention OFF -> follow-ups get 0 reuse. ----
        var (offA, offB) = await RunTwoRoundsAsync(retentionEnabled: false);
        Assert.Equal(0, offA.PrefixCacheReusedTokens);
        Assert.Equal(0, offB.PrefixCacheReusedTokens);

        // ---- Verify the fix: retention ON -> follow-ups reuse the whole prefix. ----
        var (onA, onB) = await RunTwoRoundsAsync(retentionEnabled: true);

        // Each follow-up's prompt = round-1 (prompt+output) + a short suffix, so the
        // reused prefix must equal the entire retained conversation (well past Cap).
        Assert.True(onA.PrefixCacheReusedTokens > Cap,
            $"follow-up A reused {onA.PrefixCacheReusedTokens} tokens (expected > {Cap})");
        Assert.True(onB.PrefixCacheReusedTokens > Cap,
            $"follow-up B reused {onB.PrefixCacheReusedTokens} tokens (expected > {Cap})");
        Assert.Equal(onA.PromptTokenCount - SuffixLen, onA.PrefixCacheReusedTokens);
        Assert.Equal(onB.PromptTokenCount - SuffixLen, onB.PrefixCacheReusedTokens);

        // High-performance check: reuse ratio is near-total (only the short new
        // suffix is re-prefilled), i.e. the multi-turn follow-up no longer pays to
        // recompute the whole conversation.
        double pctA = 100.0 * onA.PrefixCacheReusedTokens / onA.PromptTokenCount;
        double pctB = 100.0 * onB.PrefixCacheReusedTokens / onB.PromptTokenCount;
        Assert.True(pctA >= 80.0, $"follow-up A reuse {pctA:F1}% too low");
        Assert.True(pctB >= 80.0, $"follow-up B reuse {pctB:F1}% too low");
    }

    [Fact]
    public async Task ExplicitCapability_AllowsQwenLikeExactRetainedPrefix()
    {
        // Qwen 3.5/3.6 cannot share its block snapshots across requests and its
        // recurrent state cannot rewind. Its complete request-owned holder can
        // nevertheless be re-keyed when the next prompt extends it EXACTLY. In
        // particular, MaxReusablePrefixTokens is unrelated to that holder and
        // must not be used as the retained-cache capability gate.
        var (a, b) = await RunTwoRoundsAsync(
            retentionEnabled: true,
            createModel: () => new FusedStubModel(
                supportsKvCacheTruncation: false,
                supportsCrossSequenceKvReuse: false,
                maxReusablePrefixTokens: int.MaxValue,
                supportsRetainedFusedCache: true));

        Assert.Equal(a.PromptTokenCount - SuffixLen, a.PrefixCacheReusedTokens);
        Assert.Equal(b.PromptTokenCount - SuffixLen, b.PrefixCacheReusedTokens);
        Assert.True(a.PrefixCacheReusedTokens > Cap);
        Assert.True(b.PrefixCacheReusedTokens > Cap);
    }

    [Fact]
    public async Task QwenLikeNonTruncatableHolder_RejectsOmittedTail()
    {
        // EOS is forwarded into the holder but omitted from rendered history.
        // Gemma may rewind that one-token tail; a Qwen-like recurrent holder may
        // not. The retained candidate must be declined rather than rebound and
        // then passed to the model's unsupported TruncateKVCache path.
        var model = new FusedStubModel(
            peakIsEos: true,
            supportsKvCacheTruncation: false,
            supportsCrossSequenceKvReuse: false,
            maxReusablePrefixTokens: int.MaxValue,
            supportsRetainedFusedCache: true);
        var (a, b) = await RunTwoRoundsAsync(
            retentionEnabled: true,
            followUpSuffixToken: PeakToken + 1,
            createModel: () => model);

        Assert.Equal(0, a.PrefixCacheReusedTokens);
        Assert.Equal(0, b.PrefixCacheReusedTokens);
        Assert.Empty(model.TruncationTargets);
    }

    [Fact]
    public async Task WrappedRetainedHolder_RejectsUnsafeOmittedTailBeforeBinding()
    {
        var model = new FusedStubModel(
            peakIsEos: true,
            supportsCrossSequenceKvReuse: false,
            refuseWrappedRewind: true);
        var (a, b) = await RunTwoRoundsAsync(
            retentionEnabled: true,
            followUpSuffixToken: PeakToken + 1,
            createModel: () => model);

        Assert.Equal(0, a.PrefixCacheReusedTokens);
        Assert.Equal(0, b.PrefixCacheReusedTokens);
        Assert.Empty(model.TruncationTargets);
    }

    [Fact]
    public async Task RetainedRewindRefusedAtExecution_ReprefillsAndRetractsReuse()
    {
        await WithCheckpointsOnAsync(async () =>
        {
            var model = new FusedStubModel(
                peakIsEos: true,
                supportsCrossSequenceKvReuse: false,
                refuseTruncationAtExecution: true) { SupportsPrefixCheckpoints = false };
            var gate = new ComputeGate();
            using var engine = new InferenceEngine(model, Config(), NullLogger.Instance) { ComputeGate = gate };

            async Task<InferenceCompletion[]> RunPair(string round, bool appendSuffix)
            {
                SequenceState Request(int branch)
                {
                    var prompt = Enumerable.Repeat(branch, PromptLen).ToList();
                    if (appendSuffix) prompt.AddRange(Enumerable.Repeat(PeakToken + 1, SuffixLen));
                    return new SequenceState(round + branch, prompt, 4, BlockSize, SamplingConfig.Greedy);
                }
                gate.Close();
                long holds = engine.StepsHeldByGate;
                var a = engine.SubmitRequest(Request(1));
                using (var timeout = new System.Threading.CancellationTokenSource(TimeSpan.FromSeconds(5)))
                    while (engine.StepsHeldByGate == holds) await Task.Delay(1, timeout.Token);
                var b = engine.SubmitRequest(Request(2));
                gate.Open();
                var ra = DrainAsync(a);
                var rb = DrainAsync(b);
                await Task.WhenAll(ra, rb);
                return new[] { (await ra).completion, (await rb).completion };
            }

            await RunPair("dynamic-first-", appendSuffix: false);
            Assert.True(model.HasRetainedHolder("dynamic-first-1"));
            Assert.True(model.HasRetainedHolder("dynamic-first-2"));
            var second = await RunPair("dynamic-follow-", appendSuffix: true);
            Assert.Equal(2, model.ExecutionTruncationRefusals);
            Assert.Empty(model.TruncationTargets);
            foreach (var completion in second)
            {
                Assert.Equal(SequenceStatus.FinishedStopped, completion.Status);
                Assert.Equal("eos", completion.FinishReason);
                Assert.Equal(0, completion.PrefixCacheReusedTokens);
                Assert.Equal(PromptLen + SuffixLen, completion.PromptTokenCount);
            }
            foreach (string requestId in new[] { "dynamic-follow-1", "dynamic-follow-2" })
                Assert.Contains(model.ForwardCalls, call => call.RequestId == requestId
                    && call.Start == 0 && call.Count == PromptLen + SuffixLen);
        });
    }

    [Fact]
    public async Task FiniteSnapshotCap_WithoutExplicitCapability_DoesNotRetain()
    {
        // The old gate inferred retained-holder support from a finite pooled
        // snapshot cap. Keep the two concepts independent: a model must opt in
        // to the retain/re-key/discard lifecycle explicitly.
        var (a, b) = await RunTwoRoundsAsync(
            retentionEnabled: true,
            createModel: () => new FusedStubModel(supportsRetainedFusedCache: false));

        Assert.Equal(0, a.PrefixCacheReusedTokens);
        Assert.Equal(0, b.PrefixCacheReusedTokens);
    }

    [Fact]
    public void AllDecodeBatchedEarlyReturn_StillTracksSequencesForRetention()
    {
        string previousRetention = Environment.GetEnvironmentVariable("TS_RETAINED_FUSED_CACHE");
        string previousBudget = Environment.GetEnvironmentVariable("TS_RETAINED_FUSED_CACHE_MAX");
        string previousBatched = Environment.GetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED");
        string previousPerSeq = Environment.GetEnvironmentVariable("TS_PER_SEQ_FUSED");
        string previousTokenBatch = Environment.GetEnvironmentVariable("TS_BATCHED_FUSED_DECODE");
        Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE", "1");
        Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE_MAX", "4");
        Environment.SetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED", "0");
        Environment.SetEnvironmentVariable("TS_PER_SEQ_FUSED", "1");
        Environment.SetEnvironmentVariable("TS_BATCHED_FUSED_DECODE", "1");
        try
        {
            var model = new FusedStubModel(
                supportsKvCacheTruncation: false,
                supportsCrossSequenceKvReuse: false,
                maxReusablePrefixTokens: int.MaxValue,
                supportsRetainedFusedCache: true,
                batchedFusedDecodeSucceeds: true);
            var cfg = Config();
            var pool = new BlockPool(
                cfg.NumBlocks, cfg.BlockSize, model.ComputeKVBlockByteSize(cfg.BlockSize));
            var scheduler = new ContinuousBatchScheduler(
                cfg,
                pool,
                model.KVStateFingerprint,
                NullLogger.Instance,
                supportsCrossSequenceKvReuse: model.SupportsCrossSequenceKvReuse,
                maxReusablePrefixTokens: model.MaxReusablePrefixTokens);
            var executor = new BatchExecutor(model, pool, scheduler, NullLogger.Instance);

            SequenceState PrimeDecodeSequence(string requestId, int promptToken)
            {
                var prompt = Enumerable.Repeat(promptToken, PromptLen).ToList();
                var seq = new SequenceState(
                    requestId, prompt, maxNewTokens: 1, BlockSize, SamplingConfig.Greedy);
                // Include the one-token decode scheduled below; unlike the real
                // scheduler, this direct executor test must reserve that capacity.
                var blocks = pool.AllocateNew((PromptLen + 1 + BlockSize - 1) / BlockSize)
                    ?? throw new InvalidOperationException("test block pool exhausted");
                foreach (var block in blocks)
                    seq.BlockTable.AppendBlock(block);

                Assert.True(model.BindSequenceCache(requestId));
                seq.LastLogits = model.Forward(prompt.ToArray());
                seq.AdvanceComputedTokens(PromptLen);
                seq.Status = SequenceStatus.Running;
                return seq;
            }

            var a = PrimeDecodeSequence("batched-retain-a", 1);
            var b = PrimeDecodeSequence("batched-retain-b", 2);
            model.RestorePrimaryCache();

            var step = new SchedulerOutput();
            step.ScheduledWork.Add(new ScheduledSequenceWork(a, 1, isNewAdmission: false, isPrefill: false));
            step.ScheduledWork.Add(new ScheduledSequenceWork(b, 1, isNewAdmission: false, isPrefill: false));

            var results = executor.ExecuteStep(step);

            Assert.Equal(2, results.Count);
            Assert.All(results, result => Assert.Null(result.Error));
            Assert.Equal(1, model.SuccessfulBatchedFusedDecodeCalls);

            // ExecuteStepPerSequenceFused returns immediately when the whole
            // decode set succeeds in one batched call. Tracking must happen
            // before that return or clean release cannot retain either holder.
            a.Status = SequenceStatus.FinishedLengthCapped;
            b.Status = SequenceStatus.FinishedLengthCapped;
            Assert.True(executor.TryRetainReleasedFusedCache(a.RequestId));
            Assert.True(executor.TryRetainReleasedFusedCache(b.RequestId));
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE", previousRetention);
            Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE_MAX", previousBudget);
            Environment.SetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED", previousBatched);
            Environment.SetEnvironmentVariable("TS_PER_SEQ_FUSED", previousPerSeq);
            Environment.SetEnvironmentVariable("TS_BATCHED_FUSED_DECODE", previousTokenBatch);
        }
    }

    [Fact]
    public async Task ExplicitFollowUpBoundary_DoesNotReuseCompleteRetainedFusedHolder()
    {
        var (a, b) = await RunTwoRoundsAsync(
            retentionEnabled: true,
            followUpCacheBoundary: BlockSize);

        Assert.True(a.PrefixCacheReusedTokens <= BlockSize,
            $"follow-up A reused {a.PrefixCacheReusedTokens} tokens past its explicit boundary");
        Assert.True(b.PrefixCacheReusedTokens <= BlockSize,
            $"follow-up B reused {b.PrefixCacheReusedTokens} tokens past its explicit boundary");
    }

    [Fact]
    public async Task ExplicitSourceBoundary_DoesNotRetainCompleteFusedHolder()
    {
        var (a, b) = await RunTwoRoundsAsync(
            retentionEnabled: true,
            firstRoundCacheBoundary: BlockSize);

        // Per-sequence fused execution does not publish pooled snapshots. With
        // source-side retention correctly vetoed, no cross-request prefix remains.
        Assert.Equal(0, a.PrefixCacheReusedTokens);
        Assert.Equal(0, b.PrefixCacheReusedTokens);
    }

    [Fact]
    public async Task DifferentMediaAtTheStart_DoesNotReusePlaceholderIdenticalRetainedHolder()
    {
        var (a, b) = await RunTwoRoundsAsync(
            retentionEnabled: true,
            firstRoundMedia: ImageAt(0, "img:a"),
            followUpMedia: ImageAt(0, "img:b"));

        Assert.Equal(0, a.PrefixCacheReusedTokens);
        Assert.Equal(0, b.PrefixCacheReusedTokens);
    }

    [Fact]
    public async Task OmittedEos_RewindsRetainedHolderBeforeContinuing()
    {
        string previous = Environment.GetEnvironmentVariable("TS_RETAINED_FUSED_CACHE");
        Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE", "1");
        try
        {
            var model = new FusedStubModel(peakIsEos: true);
            using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);

            var promptA = Enumerable.Repeat(1, PromptLen).ToList();
            var promptB = Enumerable.Repeat(2, PromptLen).ToList();
            var hA1 = engine.SubmitRequest(new SequenceState(
                "eos-A1", promptA, 4, BlockSize, SamplingConfig.Greedy));
            var hB1 = engine.SubmitRequest(new SequenceState(
                "eos-B1", promptB, 4, BlockSize, SamplingConfig.Greedy));
            var rA1 = DrainAsync(hA1);
            var rB1 = DrainAsync(hB1);
            await Task.WhenAll(rA1, rB1);
            var firstA = await rA1;
            var firstB = await rB1;

            // The engine forwards EOS into K/V but deliberately does not publish it,
            // so the next rendered history extends the visible prompt, not the full
            // retained holder token run.
            Assert.Empty(firstA.output);
            Assert.Empty(firstB.output);

            var followA = new List<int>(promptA);
            followA.AddRange(Enumerable.Repeat(PeakToken + 1, SuffixLen));
            var followB = new List<int>(promptB);
            followB.AddRange(Enumerable.Repeat(PeakToken + 1, SuffixLen));
            var hA2 = engine.SubmitRequest(new SequenceState(
                "eos-A2", followA, 4, BlockSize, SamplingConfig.Greedy));
            var hB2 = engine.SubmitRequest(new SequenceState(
                "eos-B2", followB, 4, BlockSize, SamplingConfig.Greedy));
            var rA2 = DrainAsync(hA2);
            var rB2 = DrainAsync(hB2);
            await Task.WhenAll(rA2, rB2);
            var secondA = await rA2;
            var secondB = await rB2;

            Assert.Equal(PromptLen, secondA.completion.PrefixCacheReusedTokens);
            Assert.Equal(PromptLen, secondB.completion.PrefixCacheReusedTokens);
            Assert.Equal(2, model.TruncationTargets.Count(t => t == PromptLen));
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE", previous);
        }
    }

    // ---------------------------------------------------------------------------
    // Shared-prefix checkpoints: the prompt every conversation begins with is
    // copied once and every later new chat starts from a clone of the copy.
    // ---------------------------------------------------------------------------

    private const int SharedPrefixLen = 48;
    // Longer than the live-cache rewind allowance (16), so a new chat cannot be
    // served by rewinding the previous chat's live cache and must use the checkpoint.
    private const int FirstMessageLen = 20;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task PrefixMetadataAllocationFailure_DoesNotCreateAnUntrackedModelCopy(bool fromStore)
    {
        await WithCheckpointsOnAsync(() =>
        {
            var model = new FusedStubModel();
            var executor = NewRetentionExecutor(model);
            try
            {
                var seq = NewChat("prefix-allocation", 5);
                if (fromStore)
                {
                    var store = new MemoryCheckpointStore();
                    store.Save(model.KVStateFingerprint + "|prefix-checkpoint-v1", SharedPrefix().ToArray(), stream =>
                    {
                        using var writer = new System.IO.BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);
                        writer.Write(0x53545542u);
                        writer.Write(SharedPrefixLen);
                    });
                    executor.PrefixCheckpointStore = store;
                }
                else
                {
                    model.BindSequenceCache(seq.RequestId);
                    model.Forward(seq.PromptTokens.Take(SharedPrefixLen).ToArray());
                    AdvancePrefixCheckpointFixture(executor, seq);
                }
                executor.AllocateRetainedCacheTokens = _ => throw new OutOfMemoryException("controlled prefix metadata allocation");
                if (fromStore) Assert.Equal(0, executor.ComputeFusedContinuationLcp(seq));
                else InvokePrefixMethod(executor, "MaybeCheckpointSharedPrefix", seq);
                Assert.Empty(PrefixMetadata(executor));
                Assert.Empty(model.Checkpoints);
                Assert.Equal(0, model.Imports);

                executor.AllocateRetainedCacheTokens = n => new int[n];
                if (fromStore) Assert.Equal(SharedPrefixLen, executor.ComputeFusedContinuationLcp(seq));
                else
                {
                    var retry = NewChat("prefix-allocation-retry", 6);
                    AdvancePrefixCheckpointFixture(executor, retry);
                    InvokePrefixMethod(executor, "MaybeCheckpointSharedPrefix", retry);
                }
                var entry = Assert.Single(PrefixMetadata(executor));
                Assert.True(model.HasRetainedHolder(entry.Key));
                Assert.Equal(SharedPrefix().ToArray(), entry.Tokens);
            }
            finally { executor.Reset(); }
            return Task.CompletedTask;
        });
    }

    [Fact]
    public async Task ExistingPrefixRefresh_ReusesItsOwnershipNode()
    {
        await WithCheckpointsOnAsync(() =>
        {
            var model = new FusedStubModel();
            var executor = NewRetentionExecutor(model);
            try
            {
                var seq = PrimePrefixCheckpoint(executor, model, "first", 1);
                object node = PrefixFirstNode(executor);
                executor.AllocateRetainedCacheTokens = _ => throw new OutOfMemoryException("refresh must not allocate metadata");
                var identical = NewChat("same-prefix", 5);
                AdvancePrefixCheckpointFixture(executor, identical);
                InvokePrefixMethod(executor, "MaybeCheckpointSharedPrefix", identical);
                Assert.Same(node, PrefixFirstNode(executor));
                Assert.Single(model.Checkpoints);
                Assert.True(model.HasRetainedHolder(Assert.Single(PrefixMetadata(executor)).Key));
            }
            finally { executor.Reset(); }
            return Task.CompletedTask;
        });
    }

    [Fact]
    public async Task PrefixDiscardFailure_RetainsOwnershipUntilSuccessfulRetry()
    {
        await WithCheckpointsOnAsync(() =>
        {
            var model = new FusedStubModel();
            var executor = NewRetentionExecutor(model);
            try
            {
                PrimePrefixCheckpoint(executor, model, "first", 1);
                string old = Assert.Single(PrefixMetadata(executor)).Key;
                model.ThrowOnDiscardKey = old;
                Assert.Throws<InvalidOperationException>(() => PrimePrefixCheckpoint(executor, model, "second", 2));
                Assert.Equal(2, PrefixMetadata(executor).Length);
                Assert.All(PrefixMetadata(executor), entry => Assert.True(model.HasRetainedHolder(entry.Key)));
                Assert.Empty(model.DiscardedRetainedRequestIds);
                model.ThrowOnDiscardKey = null;
                InvokePrefixMethod(executor, "EvictPrefixCheckpointsBeyondBudget", model);
                Assert.Single(PrefixMetadata(executor));
                Assert.False(model.HasRetainedHolder(old));
                Assert.Equal(new[] { old }, model.DiscardedRetainedRequestIds);
            }
            finally { model.ThrowOnDiscardKey = null; executor.Reset(); }
            return Task.CompletedTask;
        }, budget: "1");
    }

    private static SequenceState PrimePrefixCheckpoint(BatchExecutor executor, FusedStubModel model, string id, int token)
    {
        var seq = new SequenceState(id, Enumerable.Repeat(token, SharedPrefixLen + FirstMessageLen).ToList(),
            1, BlockSize, SamplingConfig.Greedy, sharedPrefixTokens: SharedPrefixLen);
        model.BindSequenceCache(id);
        model.Forward(seq.PromptTokens.Take(SharedPrefixLen).ToArray());
        AdvancePrefixCheckpointFixture(executor, seq);
        InvokePrefixMethod(executor, "MaybeCheckpointSharedPrefix", seq);
        return seq;
    }

    private static void AdvancePrefixCheckpointFixture(BatchExecutor executor, SequenceState seq)
    {
        var blocks = ExecutorPool(executor).AllocateNew(SharedPrefixLen / BlockSize)
            ?? throw new InvalidOperationException("prefix fixture block pool exhausted");
        foreach (var block in blocks) seq.BlockTable.AppendBlock(block);
        seq.AdvanceComputedTokens(SharedPrefixLen);
    }

    private static object PrefixList(BatchExecutor executor) => typeof(BatchExecutor).GetField("_prefixCheckpoints",
        System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.GetValue(executor)!;

    private static object PrefixFirstNode(BatchExecutor executor)
    {
        object list = PrefixList(executor);
        return list.GetType().GetProperty("First")!.GetValue(list)!;
    }

    private static (string Key, int[] Tokens)[] PrefixMetadata(BatchExecutor executor) =>
        ((System.Collections.IEnumerable)PrefixList(executor)).Cast<object>().Select(entry => (
            (string)entry.GetType().GetField("RequestId")!.GetValue(entry)!,
            (int[])entry.GetType().GetField("Tokens")!.GetValue(entry)!)).ToArray();

    private static void InvokePrefixMethod(BatchExecutor executor, string name, object argument)
    {
        try
        {
            typeof(BatchExecutor).GetMethod(name, System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!
                .Invoke(executor, new[] { argument });
        }
        catch (System.Reflection.TargetInvocationException ex) when (ex.InnerException != null)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
            throw;
        }
    }

    private static List<int> SharedPrefix() => Enumerable.Repeat(1, SharedPrefixLen).ToList();

    /// <summary>Every switch these tests depend on, set explicitly: other test classes
    /// flip the same variables while running in parallel.</summary>
    private static async Task WithCheckpointsOnAsync(Func<Task> body, string budget = null)
    {
        string prevRetained = Environment.GetEnvironmentVariable("TS_RETAINED_FUSED_CACHE");
        string prevPerSeq = Environment.GetEnvironmentVariable("TS_PER_SEQ_FUSED");
        string prevCheckpoints = Environment.GetEnvironmentVariable("TS_PREFIX_CHECKPOINTS");
        string prevBudget = Environment.GetEnvironmentVariable("TS_PREFIX_CHECKPOINTS_MAX");
        Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE", "1");
        Environment.SetEnvironmentVariable("TS_PER_SEQ_FUSED", "1");
        Environment.SetEnvironmentVariable("TS_PREFIX_CHECKPOINTS", "1");
        Environment.SetEnvironmentVariable("TS_PREFIX_CHECKPOINTS_MAX", budget);
        try { await body(); }
        finally
        {
            Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE", prevRetained);
            Environment.SetEnvironmentVariable("TS_PER_SEQ_FUSED", prevPerSeq);
            Environment.SetEnvironmentVariable("TS_PREFIX_CHECKPOINTS", prevCheckpoints);
            Environment.SetEnvironmentVariable("TS_PREFIX_CHECKPOINTS_MAX", prevBudget);
        }
    }

    private static SequenceState NewChat(string id, int firstToken, int count = FirstMessageLen, int sharedPrefix = SharedPrefixLen)
    {
        var prompt = SharedPrefix();
        prompt.AddRange(Enumerable.Repeat(firstToken, count));
        return new SequenceState(id, prompt, 4, BlockSize, SamplingConfig.Greedy, sharedPrefixTokens: sharedPrefix);
    }

    [Fact]
    public async Task SharedPrefix_IsCheckpointedOnceAndClonedForEveryNewChat()
    {
        await WithCheckpointsOnAsync(async () =>
        {
            var model = new FusedStubModel();
            using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);

            // The first chat: its prefill stops exactly at the shared prefix (the
            // solo chunk would otherwise swallow the whole 51-token prompt in one
            // pass), the model's state is copied there, and the chat carries on.
            var first = await DrainAsync(engine.SubmitRequest(NewChat("chat-1", firstToken: 7)));
            Assert.Equal(0, first.completion.PrefixCacheReusedTokens);
            Assert.Single(model.Checkpoints);
            Assert.Equal(SharedPrefixLen, model.Checkpoints.Values.Single());
            Assert.Equal(0, model.Clones);

            // A second chat with a different first message: no live cache to
            // continue (it diverges right after the prefix) and no retained
            // conversation matches, so it is the clone that serves it - the whole
            // shared prefix reused, only the new message forwarded.
            var second = await DrainAsync(engine.SubmitRequest(NewChat("chat-2", firstToken: 8)));
            Assert.Equal(SharedPrefixLen, second.completion.PrefixCacheReusedTokens);
            Assert.Equal(1, model.Clones);
            Assert.Single(model.Checkpoints);   // still there: cloned, not consumed

            // And a third, and no second checkpoint of the same prefix.
            var third = await DrainAsync(engine.SubmitRequest(NewChat("chat-3", firstToken: 9)));
            Assert.Equal(SharedPrefixLen, third.completion.PrefixCacheReusedTokens);
            Assert.Equal(2, model.Clones);
            Assert.Single(model.Checkpoints);
            Assert.Empty(model.DiscardedRetainedRequestIds);
        });
    }

    [Fact]
    public async Task SpeculationRunsOnARetainedHolder_AndTheStreamIsWhatPlainDecodingGives()
    {
        // Every turn after a chat's first lives in a per-request fused holder, and
        // the planner sends those to the per-sequence fused path, where speculation
        // used to be impossible ("sequence lives in a per-request fused cache"). The
        // follow-up here continues the retained holder of round one and must still
        // draft, verify in batches, and emit the plain stream.
        string prevRetained = Environment.GetEnvironmentVariable("TS_RETAINED_FUSED_CACHE");
        string prevPerSeq = Environment.GetEnvironmentVariable("TS_PER_SEQ_FUSED");
        Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE", "1");
        Environment.SetEnvironmentVariable("TS_PER_SEQ_FUSED", "1");
        try
        {
            SpeculationOptions ngram = new()
            {
                Enabled = true,
                SpeculatorName = TensorSharp.Runtime.Speculative.SpeculatorRegistry.NGram,
                MaxDraftTokens = 4,
            };

            // The same conversation twice: plain, and speculative. Round one is a
            // CONCURRENT pair, which is what puts each conversation in its own fused
            // holder; the follow-up is then solo and continues its retained holder.
            async Task<(List<int> output, SequenceState seq, FusedStubModel model, int contexts)> RunAsync(bool speculative)
            {
                var model = new FusedStubModel(periodicPeak: true);
                SchedulerConfig cfg = speculative ? Config().WithSpeculation(ngram) : Config();
                using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);

                var promptA = Enumerable.Repeat(1, PromptLen).ToList();
                var promptB = Enumerable.Repeat(2, PromptLen).ToList();
                var hA = engine.SubmitRequest(new SequenceState("spec-A1", promptA, Round1NewTokens, BlockSize, SamplingConfig.Greedy));
                var hB = engine.SubmitRequest(new SequenceState("spec-B1", promptB, Round1NewTokens, BlockSize, SamplingConfig.Greedy));
                var rA = DrainAsync(hA);
                var rB = DrainAsync(hB);
                await Task.WhenAll(rA, rB);
                var (_, outA) = await rA;

                var follow = new List<int>(promptA);
                follow.AddRange(outA);
                follow.AddRange(Enumerable.Repeat(PeakToken + 1, SuffixLen));
                var seq = new SequenceState("spec-A2", follow, 24, BlockSize, SamplingConfig.Greedy);
                var (completion, output) = await DrainAsync(engine.SubmitRequest(seq));
                Assert.Equal(PromptLen + Round1NewTokens, completion.PrefixCacheReusedTokens);
                // The context is dropped when the engine thread retains the holder,
                // a moment after the completion reached us.
                int contexts = FusedSpecContexts(engine);
                for (int i = 0; i < 200 && contexts != 0; i++)
                {
                    await Task.Delay(10);
                    contexts = FusedSpecContexts(engine);
                }
                return (output, seq, model, contexts);
            }

            var plain = await RunAsync(speculative: false);
            var spec = await RunAsync(speculative: true);

            Assert.Equal(24, plain.output.Count);
            Assert.Equal(plain.output, spec.output);
            Assert.Null(plain.seq.SpecStats);
            Assert.NotNull(spec.seq.SpecStats);
            Assert.True(spec.seq.SpecStats.VerifySteps > 0, "speculation never verified a window on the fused holder");
            Assert.True(spec.seq.SpecStats.TokensAccepted > 0);
            Assert.True(spec.model.SpecForwardCalls > 0);
            // The holder is retained for the next turn; the speculative context is not.
            Assert.Equal(0, spec.contexts);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE", prevRetained);
            Environment.SetEnvironmentVariable("TS_PER_SEQ_FUSED", prevPerSeq);
        }
    }

    [Fact]
    public async Task ANeighbourArrivingMidStream_LeavesTheSpeculatingRequestsStreamUnchanged()
    {
        // A solo request speculates on its holder; a second request lands while it
        // decodes, so the step turns mixed (plain for both, batched) and turns solo
        // again when the neighbour finishes. The first request's stream must be
        // what plain decoding gives, the transition must re-arm speculation, and
        // no speculative context may outlive its request.
        string prevRetained = Environment.GetEnvironmentVariable("TS_RETAINED_FUSED_CACHE");
        string prevPerSeq = Environment.GetEnvironmentVariable("TS_PER_SEQ_FUSED");
        Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE", "1");
        Environment.SetEnvironmentVariable("TS_PER_SEQ_FUSED", "1");
        try
        {
            SpeculationOptions ngram = new()
            {
                Enabled = true,
                SpeculatorName = TensorSharp.Runtime.Speculative.SpeculatorRegistry.NGram,
                MaxDraftTokens = 4,
            };
            const int newTokens = 96;
            var promptA = Enumerable.Repeat(1, PromptLen).ToList();
            var promptB = Enumerable.Repeat(2, PromptLen).ToList();

            async Task<(List<int> a, SequenceState seqA, SequenceState seqB, int contexts)> RunAsync(bool speculative)
            {
                var model = new FusedStubModel(periodicPeak: true);
                SchedulerConfig cfg = speculative ? Config().WithSpeculation(ngram) : Config();
                using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);
                var seqA = new SequenceState(speculative ? "stag-A-spec" : "stag-A-plain", promptA, newTokens, BlockSize, SamplingConfig.Greedy);
                var hA = engine.SubmitRequest(seqA);
                // Let A decode a while alone (its probe runs here), then admit B.
                var a = new List<int>();
                while (a.Count < 12)
                    a.Add(await hA.Tokens.ReadAsync());
                var statsBefore = seqA.SpecStats;
                var seqB = new SequenceState(speculative ? "stag-B-spec" : "stag-B-plain", promptB, 24, BlockSize, SamplingConfig.Greedy);
                var (_, _) = await DrainAsync(engine.SubmitRequest(seqB));
                await foreach (var t in hA.Tokens.ReadAllAsync())
                    a.Add(t);
                await hA.Completion;
                // The interlude must not have re-armed A from scratch: the same
                // execution (and its stats) carries on once A is the only decoder.
                if (speculative)
                    Assert.Same(statsBefore, seqA.SpecStats);
                int contexts = FusedSpecContexts(engine);
                for (int i = 0; i < 200 && contexts != 0; i++) { await Task.Delay(10); contexts = FusedSpecContexts(engine); }
                return (a, seqA, seqB, contexts);
            }

            var plain = await RunAsync(speculative: false);
            var spec = await RunAsync(speculative: true);
            Assert.Equal(newTokens, plain.a.Count);
            Assert.Equal(plain.a, spec.a);
            Assert.NotNull(spec.seqA.SpecStats);
            Assert.True(spec.seqA.SpecStats.VerifySteps > 0, "A never verified a window across its solo phases");
            Assert.Equal(0, spec.contexts);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE", prevRetained);
            Environment.SetEnvironmentVariable("TS_PER_SEQ_FUSED", prevPerSeq);
        }
    }

    private static int FusedSpecContexts(InferenceEngine engine)
    {
        const System.Reflection.BindingFlags flags =
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        object executor = typeof(InferenceEngine).GetField("_executor", flags)!.GetValue(engine)!;
        var contexts = (System.Collections.IDictionary)executor.GetType().GetField("_fusedSpecCtx", flags)!.GetValue(executor)!;
        return contexts.Count;
    }

    /// <summary>A store that keeps checkpoints in memory, keyed exactly as a file store would.</summary>
    private sealed class MemoryCheckpointStore : IPrefixCheckpointStore
    {
        private readonly Dictionary<string, byte[]> _files = new(StringComparer.Ordinal);
        public int Saves { get; private set; }
        public int Opens { get; private set; }
        public int Count => _files.Count;
        public bool Corrupt { get; set; }

        private static string KeyFor(string fp, ReadOnlySpan<int> tokens) => fp + "|" + string.Join(",", tokens.ToArray());

        public bool TryOpen(string modelFingerprint, ReadOnlySpan<int> prefixTokens, out System.IO.Stream payload)
        {
            payload = null;
            if (!_files.TryGetValue(KeyFor(modelFingerprint, prefixTokens), out byte[] bytes)) return false;
            Opens++;
            payload = new System.IO.MemoryStream(Corrupt ? new byte[] { 9, 9, 9, 9, 9, 9, 9, 9 } : bytes, writable: false);
            return true;
        }

        public bool Save(string modelFingerprint, ReadOnlySpan<int> prefixTokens, Action<System.IO.Stream> writePayload)
        {
            var ms = new System.IO.MemoryStream();
            writePayload(ms);
            _files[KeyFor(modelFingerprint, prefixTokens)] = ms.ToArray();
            Saves++;
            return true;
        }
    }

    [Fact]
    public async Task ACheckpointSavedByOneProcess_ServesTheFirstChatOfTheNext()
    {
        await WithCheckpointsOnAsync(async () =>
        {
            var store = new MemoryCheckpointStore();

            // Process 1: the first chat crosses the shared prefix, the checkpoint is
            // taken in memory AND written to the store.
            var first = new FusedStubModel();
            using (var engine = new InferenceEngine(first, Config(), NullLogger.Instance) { PrefixCheckpointStore = store })
            {
                var chat = await DrainAsync(engine.SubmitRequest(NewChat("p1-chat-1", firstToken: 7)));
                Assert.Equal(0, chat.completion.PrefixCacheReusedTokens);
                Assert.Single(first.Checkpoints);
                Assert.Equal(1, first.Exports);
                Assert.Equal(1, store.Saves);
            }

            // Process 2: a fresh model, nothing in memory. Its very first chat is served
            // from the store -- the whole prefix reused, no checkpoint prefilled, one
            // read -- and the chat after it from the same restored copy.
            var second = new FusedStubModel();
            using (var engine = new InferenceEngine(second, Config(), NullLogger.Instance) { PrefixCheckpointStore = store })
            {
                var chat = await DrainAsync(engine.SubmitRequest(NewChat("p2-chat-1", firstToken: 8)));
                Assert.Equal(SharedPrefixLen, chat.completion.PrefixCacheReusedTokens);
                Assert.Equal(1, second.Imports);
                Assert.Equal(1, second.Clones);
                Assert.Empty(second.Checkpoints);      // restored, never prefilled
                Assert.Equal(1, store.Opens);

                var next = await DrainAsync(engine.SubmitRequest(NewChat("p2-chat-2", firstToken: 9)));
                Assert.Equal(SharedPrefixLen, next.completion.PrefixCacheReusedTokens);
                Assert.Equal(1, second.Imports);       // in memory now: no second read
                Assert.Equal(1, store.Opens);
                Assert.Equal(2, second.Clones);
            }
            Assert.Equal(1, store.Saves);              // nothing new to save in process 2
        });
    }

    [Fact]
    public async Task AStoredCheckpointTheModelRejects_IsPrefilledAndSavedAgain()
    {
        await WithCheckpointsOnAsync(async () =>
        {
            var store = new MemoryCheckpointStore();
            var first = new FusedStubModel();
            using (var engine = new InferenceEngine(first, Config(), NullLogger.Instance) { PrefixCheckpointStore = store })
                await DrainAsync(engine.SubmitRequest(NewChat("p1-chat-1", firstToken: 7)));
            Assert.Equal(1, store.Saves);

            store.Corrupt = true;
            var second = new FusedStubModel();
            using (var engine = new InferenceEngine(second, Config(), NullLogger.Instance) { PrefixCheckpointStore = store })
            {
                var chat = await DrainAsync(engine.SubmitRequest(NewChat("p2-chat-1", firstToken: 8)));
                // Not served from the bad bytes: prefilled like a cold chat, the
                // checkpoint taken in memory, and the store given a good copy again.
                Assert.Equal(0, chat.completion.PrefixCacheReusedTokens);
                Assert.Equal(0, second.Imports);
                Assert.Single(second.Checkpoints);
                Assert.Equal(2, store.Saves);
            }
        });
    }

    [Fact]
    public async Task AFollowUpTurn_ContinuesItsOwnConversationRatherThanTheCheckpoint()
    {
        await WithCheckpointsOnAsync(async () =>
        {
            var model = new FusedStubModel();
            using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);

            var first = await DrainAsync(engine.SubmitRequest(NewChat("chat-1", firstToken: 7)));
            Assert.Single(model.Checkpoints);

            // The same conversation, one turn later: prompt = the whole first turn
            // (prompt and answer) plus a new message. That extends the live cache
            // exactly, which is longer than the checkpoint, so it wins.
            var followUp = SharedPrefix();
            followUp.AddRange(Enumerable.Repeat(7, FirstMessageLen));
            followUp.AddRange(first.output);
            followUp.AddRange(Enumerable.Repeat(PeakToken + 1, SuffixLen));
            var second = await DrainAsync(engine.SubmitRequest(new SequenceState(
                "chat-1-turn-2", followUp, 4, BlockSize, SamplingConfig.Greedy, sharedPrefixTokens: SharedPrefixLen)));

            Assert.Equal(followUp.Count - SuffixLen, second.completion.PrefixCacheReusedTokens);
            Assert.True(second.completion.PrefixCacheReusedTokens > SharedPrefixLen);
            Assert.Equal(0, model.Clones);
            Assert.Single(model.Checkpoints);
        });
    }

    [Fact]
    public async Task ADifferentSharedPrefix_GetsItsOwnCheckpoint_AndTheBudgetEvictsTheOldest()
    {
        await WithCheckpointsOnAsync(async () =>
        {
            var model = new FusedStubModel();
            using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);

            await DrainAsync(engine.SubmitRequest(NewChat("chat-1", firstToken: 7)));
            string firstKey = model.Checkpoints.Keys.Single();

            // A different system prompt (a changed skill selection, thinking toggled
            // on a Gemma 4 template): a different prefix, its own checkpoint, and
            // with a budget of one the old one goes.
            var other = Enumerable.Repeat(2, SharedPrefixLen).ToList();
            other.AddRange(Enumerable.Repeat(7, FirstMessageLen));
            await DrainAsync(engine.SubmitRequest(new SequenceState(
                "chat-2", other, 4, BlockSize, SamplingConfig.Greedy, sharedPrefixTokens: SharedPrefixLen)));

            Assert.Equal(2, model.Checkpoints.Count);
            Assert.Contains(firstKey, model.DiscardedRetainedRequestIds);

            // The surviving prefix is cloned for its next new chat...
            int clonesBefore = model.Clones;
            var otherAgain = new List<int>(Enumerable.Repeat(2, SharedPrefixLen));
            otherAgain.AddRange(Enumerable.Repeat(8, FirstMessageLen));
            var onOther = await DrainAsync(engine.SubmitRequest(new SequenceState(
                "chat-3", otherAgain, 4, BlockSize, SamplingConfig.Greedy, sharedPrefixTokens: SharedPrefixLen)));
            Assert.Equal(SharedPrefixLen, onOther.completion.PrefixCacheReusedTokens);
            Assert.Equal(clonesBefore + 1, model.Clones);

            // ...and the evicted one no longer serves a new chat as a whole (the pooled
            // block cache may still hand back its first window, which is what it did
            // before checkpoints existed). That chat takes a fresh checkpoint of its
            // prefix, which with a budget of one evicts the other in turn.
            var backToFirst = await DrainAsync(engine.SubmitRequest(NewChat("chat-4", firstToken: 8)));
            Assert.True(backToFirst.completion.PrefixCacheReusedTokens < SharedPrefixLen);
            Assert.Equal(clonesBefore + 1, model.Clones);
            Assert.Equal(3, model.Checkpoints.Count);
            Assert.Equal(2, model.DiscardedRetainedRequestIds.Count);
        }, budget: "1");
    }

    [Fact]
    public async Task OnACircularCache_AnExactCheckpointBeatsALossyRewind()
    {
        await WithCheckpointsOnAsync(async () =>
        {
            // The stub's pooled cap (16) marks it as a sliding-window model. Chat 1's
            // message is short, so chat 2 COULD be served by rewinding chat 1's live
            // cache a few tokens - which on a circular cache reads stale keys. The
            // checkpoint serves the same prefix exactly, so it must win: a clone, and
            // no truncation of the live cache.
            var model = new FusedStubModel();
            using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);

            await DrainAsync(engine.SubmitRequest(NewChat("chat-1", firstToken: 7, count: 3)));
            Assert.Single(model.Checkpoints);

            var second = await DrainAsync(engine.SubmitRequest(NewChat("chat-2", firstToken: 8, count: 3)));
            Assert.Equal(SharedPrefixLen, second.completion.PrefixCacheReusedTokens);
            Assert.Equal(1, model.Clones);
            Assert.Empty(model.TruncationTargets);
        });
    }

    [Fact]
    public async Task WhereACloneCouldNotRun_NoCheckpointIsTaken()
    {
        // A clone lives in a per-request fused holder and runs on the fused path. With
        // that path switched off the planner would send the clone's request down the
        // linear path with placeholder blocks nothing wrote, so the checkpoint must not
        // be taken in the first place - not taken and then silently re-prefilled.
        await WithCheckpointsOnAsync(async () =>
        {
            Environment.SetEnvironmentVariable("TS_PER_SEQ_FUSED", "0");
            var model = new FusedStubModel();
            using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);
            await DrainAsync(engine.SubmitRequest(NewChat("chat-1", firstToken: 7)));
            var second = await DrainAsync(engine.SubmitRequest(NewChat("chat-2", firstToken: 8)));
            Assert.Empty(model.Checkpoints);
            Assert.Equal(0, model.Clones);
            Assert.True(second.completion.PrefixCacheReusedTokens < SharedPrefixLen);
        });
    }

    [Fact]
    public async Task AmongRetainedHolders_AnExactCheckpointBeatsARewoundOne_ButNotALongerExactConversation()
    {
        await WithCheckpointsOnAsync(async () =>
        {
            // Two chats at once, so both finish on the fused path and are retained
            // as whole-conversation holders; the shared prefix is checkpointed too.
            var model = new FusedStubModel();
            using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);
            var a = engine.SubmitRequest(NewChat("chat-a", firstToken: 7, count: 3));
            var b = engine.SubmitRequest(NewChat("chat-b", firstToken: 8, count: 3));
            var ra = DrainAsync(a);
            var rb = DrainAsync(b);
            await Task.WhenAll(ra, rb);
            var firstA = await ra;
            Assert.Single(model.Checkpoints);

            // A third, new chat: each retained conversation matches the shared prefix
            // with a short rewind (its own message and answer), the checkpoint matches
            // it exactly. On a circular cache the exact one must win — a clone, no
            // truncation — and neither conversation loses its holder.
            int clonesBefore = model.Clones;
            var third = await DrainAsync(engine.SubmitRequest(NewChat("chat-c", firstToken: 9, count: 3)));
            Assert.Equal(SharedPrefixLen, third.completion.PrefixCacheReusedTokens);
            Assert.Equal(clonesBefore + 1, model.Clones);
            Assert.Empty(model.TruncationTargets);

            // Chat A's own follow-up extends its holder exactly, which is longer than
            // the checkpoint: the conversation continues, nothing is cloned.
            var followUp = SharedPrefix();
            followUp.AddRange(Enumerable.Repeat(7, 3));
            followUp.AddRange(firstA.output);
            followUp.AddRange(Enumerable.Repeat(PeakToken + 1, SuffixLen));
            var next = await DrainAsync(engine.SubmitRequest(new SequenceState(
                "chat-a-turn-2", followUp, 4, BlockSize, SamplingConfig.Greedy, sharedPrefixTokens: SharedPrefixLen)));
            Assert.Equal(followUp.Count - SuffixLen, next.completion.PrefixCacheReusedTokens);
            Assert.Equal(clonesBefore + 1, model.Clones);
        });
    }

    [Fact]
    public async Task UnsafeLongerRetainedMatch_DoesNotHideExactCheckpointOrConsumeConversation()
    {
        await WithCheckpointsOnAsync(async () =>
        {
            var model = new FusedStubModel(refuseWrappedRewind: true);
            var gate = new ComputeGate();
            gate.Close();
            using var engine = new InferenceEngine(model, Config(), NullLogger.Instance) { ComputeGate = gate };
            var firstHandle = engine.SubmitRequest(NewChat("wrapped-a", firstToken: 7));
            using (var timeout = new System.Threading.CancellationTokenSource(TimeSpan.FromSeconds(5)))
                while (engine.StepsHeldByGate == 0) await Task.Delay(1, timeout.Token);
            var partnerHandle = engine.SubmitRequest(NewChat("wrapped-b", firstToken: 8));
            gate.Open();
            var firstTask = DrainAsync(firstHandle);
            var partnerTask = DrainAsync(partnerHandle);
            await Task.WhenAll(firstTask, partnerTask);
            var first = await firstTask;
            // Completion is published before the worker's release notification.
            // Observe retention only once that existing lifecycle hook has run.
            using (var timeout = new System.Threading.CancellationTokenSource(TimeSpan.FromSeconds(5)))
                while (!model.WasReleased("wrapped-a") || !model.WasReleased("wrapped-b"))
                    await Task.Delay(1, timeout.Token);
            Assert.True(model.HasRetainedHolder("wrapped-a"));
            Assert.Single(model.Checkpoints);

            // Match the entire first prompt but drop its output tail. The match
            // is 20 tokens longer than the exact checkpoint, exceeding the old
            // scheduler's 16-token preference allowance for an exact copy.
            // This unsafe candidate must not prevent choosing that checkpoint.
            var changed = SharedPrefix();
            changed.AddRange(Enumerable.Repeat(7, FirstMessageLen));
            changed.AddRange(Enumerable.Repeat(PeakToken + 1, SuffixLen));
            int clonesBefore = model.Clones;
            var rewritten = await DrainAsync(engine.SubmitRequest(new SequenceState(
                "wrapped-rewrite", changed, 4, BlockSize, SamplingConfig.Greedy,
                sharedPrefixTokens: SharedPrefixLen)));
            Assert.Equal(SequenceStatus.FinishedLengthCapped, rewritten.completion.Status);
            Assert.Equal(SharedPrefixLen, rewritten.completion.PrefixCacheReusedTokens);
            Assert.Equal(clonesBefore + 1, model.Clones);
            Assert.Empty(model.TruncationTargets);
            Assert.True(model.HasRetainedHolder("wrapped-a"));

            // The rejected candidate remains useful for an exact continuation.
            // This ensures the fix does not disable retention or discard a
            // valid conversation just because another request could not rewind it.
            var exact = SharedPrefix();
            exact.AddRange(Enumerable.Repeat(7, FirstMessageLen));
            exact.AddRange(first.output);
            int exactPrefix = exact.Count;
            exact.AddRange(Enumerable.Repeat(PeakToken + 2, SuffixLen));
            var continuation = await DrainAsync(engine.SubmitRequest(new SequenceState(
                "wrapped-a-next", exact, 4, BlockSize, SamplingConfig.Greedy,
                sharedPrefixTokens: SharedPrefixLen)));
            Assert.Equal(SequenceStatus.FinishedLengthCapped, continuation.completion.Status);
            Assert.Equal(exactPrefix, continuation.completion.PrefixCacheReusedTokens);
            Assert.Equal(clonesBefore + 1, model.Clones);
            Assert.Empty(model.TruncationTargets);
        });
    }

    [Fact]
    public async Task NoSharedPrefix_OrCheckpointsSwitchedOff_TakesNoCheckpoint()
    {
        await WithCheckpointsOnAsync(async () =>
        {
            var model = new FusedStubModel();
            using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);
            await DrainAsync(engine.SubmitRequest(NewChat("plain", firstToken: 7, sharedPrefix: 0)));
            Assert.Empty(model.Checkpoints);

            Environment.SetEnvironmentVariable("TS_PREFIX_CHECKPOINTS", "0");
            await DrainAsync(engine.SubmitRequest(NewChat("switched-off", firstToken: 8)));
            Assert.Empty(model.Checkpoints);
            Environment.SetEnvironmentVariable("TS_PREFIX_CHECKPOINTS", "1");

            // A model that cannot copy its state is simply never asked.
            var cannot = new FusedStubModel { SupportsPrefixCheckpoints = false };
            using var engine2 = new InferenceEngine(cannot, Config(), NullLogger.Instance);
            await DrainAsync(engine2.SubmitRequest(NewChat("cannot", firstToken: 7)));
            Assert.Empty(cannot.Checkpoints);
        });
    }

    [Fact]
    public async Task SequentialRequestIdReuse_DiscardsOldRetainedMetadataAndHolder()
    {
        string previous = Environment.GetEnvironmentVariable("TS_RETAINED_FUSED_CACHE");
        Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE", "1");
        try
        {
            var model = new FusedStubModel();
            var gate = new ComputeGate();
            gate.Close();
            using var engine = new InferenceEngine(model, Config(), NullLogger.Instance) { ComputeGate = gate };

            // Both rounds must enter the per-sequence fused path. Otherwise the
            // first request can run alone and populate legitimate pooled prefix
            // blocks, which makes the later zero-reuse assertion test a different
            // path. Wait for the worker to park before submitting the partner and
            // reopening: merely queuing two requests does not guarantee admission
            // together if the worker has already drained the first command.
            async Task WaitForParkedWorker(long previousHolds)
            {
                using var timeout = new System.Threading.CancellationTokenSource(TimeSpan.FromSeconds(5));
                while (engine.StepsHeldByGate <= previousHolds)
                    await Task.Delay(1, timeout.Token);
            }

            const string reusedId = "reused-request-id";
            var oldPrompt = Enumerable.Repeat(1, PromptLen).ToList();
            var oldPartnerPrompt = Enumerable.Repeat(2, PromptLen).ToList();
            long holds = engine.StepsHeldByGate;
            var oldHandle = engine.SubmitRequest(new SequenceState(
                reusedId, oldPrompt, 4, BlockSize, SamplingConfig.Greedy));
            await WaitForParkedWorker(holds);
            var oldPartnerHandle = engine.SubmitRequest(new SequenceState(
                "old-partner", oldPartnerPrompt, 4, BlockSize, SamplingConfig.Greedy));
            gate.Open();
            var oldResult = DrainAsync(oldHandle);
            var oldPartnerResult = DrainAsync(oldPartnerHandle);
            await Task.WhenAll(oldResult, oldPartnerResult);
            var oldConversation = await oldResult;

            // Reuse the public id for a completely unrelated conversation. Its
            // retained model holder is id-keyed, so the old metadata must be removed
            // before this new holder is stored under the same key.
            var newPrompt = Enumerable.Repeat(5, PromptLen).ToList();
            var newPartnerPrompt = Enumerable.Repeat(6, PromptLen).ToList();
            gate.Close();
            holds = engine.StepsHeldByGate;
            var newHandle = engine.SubmitRequest(new SequenceState(
                reusedId, newPrompt, 4, BlockSize, SamplingConfig.Greedy));
            await WaitForParkedWorker(holds);
            var newPartnerHandle = engine.SubmitRequest(new SequenceState(
                "new-partner", newPartnerPrompt, 4, BlockSize, SamplingConfig.Greedy));
            gate.Open();
            var newResult = DrainAsync(newHandle);
            var newPartnerResult = DrainAsync(newPartnerHandle);
            await Task.WhenAll(newResult, newPartnerResult);

            // A continuation of the OLD conversation must not match stale token
            // metadata and accidentally rebind the NEW conversation's K/V holder.
            var oldFollowUp = new List<int>(oldPrompt);
            oldFollowUp.AddRange(oldConversation.output);
            oldFollowUp.AddRange(Enumerable.Repeat(PeakToken + 1, SuffixLen));
            var followCompletion = (await DrainAsync(engine.SubmitRequest(new SequenceState(
                "old-follow-up", oldFollowUp, 4, BlockSize, SamplingConfig.Greedy)))).completion;

            Assert.Equal(0, followCompletion.PrefixCacheReusedTokens);
            Assert.Equal(1, model.DiscardedRetainedRequestIds.Count(id => id == reusedId));
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE", previous);
        }
    }

    [Fact]
    public async Task Abort_ClearsExecutorFusedBookkeepingBeforeModelReleaseCompletes()
    {
        string previous = Environment.GetEnvironmentVariable("TS_RETAINED_FUSED_CACHE");
        Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE", "1");
        try
        {
            var model = new FusedStubModel(forwardDelayMs: 1);
            using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);

            const string abortedId = "abort-fused";
            var aborted = engine.SubmitRequest(new SequenceState(
                abortedId,
                Enumerable.Repeat(1, PromptLen).ToList(),
                1000,
                BlockSize,
                SamplingConfig.Greedy));
            var partner = engine.SubmitRequest(new SequenceState(
                "abort-partner",
                Enumerable.Repeat(2, PromptLen).ToList(),
                1000,
                BlockSize,
                SamplingConfig.Greedy));

            // Seeing a decoded token proves the request passed through
            // NoteFusedSequence and has executor-owned bookkeeping to clean.
            _ = await aborted.Tokens.ReadAsync();
            Assert.True(ExecutorTracksFusedRequest(engine, abortedId));

            engine.Abort(abortedId);
            var completion = await aborted.Completion;

            Assert.Equal(SequenceStatus.FinishedAborted, completion.Status);
            Assert.True(model.WasReleased(abortedId));
            Assert.False(ExecutorTracksFusedRequest(engine, abortedId));

            engine.Abort(partner.RequestId);
            _ = await partner.Completion;

            // A Stop is how most phone turns end. The stopped sequence's holder is
            // consistent at its last forwarded token, so it is RETAINED rather than
            // freed, and the conversation's next turn — the prompt plus exactly the
            // tokens the engine forwarded — continues from it.
            var stoppedTokens = new List<int>(aborted.Sequence.PromptTokens);
            stoppedTokens.AddRange(aborted.Sequence.OutputTokens);
            var followUp = new List<int>(stoppedTokens);
            followUp.AddRange(Enumerable.Repeat(PeakToken + 1, SuffixLen));
            var next = await DrainAsync(engine.SubmitRequest(new SequenceState(
                "abort-fused-turn-2", followUp, 4, BlockSize, SamplingConfig.Greedy)));
            Assert.Equal(stoppedTokens.Count, next.completion.PrefixCacheReusedTokens);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE", previous);
        }
    }

    /// <summary>
    /// Single-stream ("请继续") analogue of the bug above, on a model whose KV cache
    /// is BLOCK-QUANTIZED (q4_0 / q8_0). Such a model declines the batched paged path
    /// — its <see cref="IBatchedPagedModel.ForwardBatch"/> throws NotSupported and it
    /// reports <c>SupportsLinearKVMigration=false</c> — but it still keeps a single
    /// live linear cache exactly like the f16 path (<see cref="FusedStubModel"/>
    /// matches this shape: ForwardBatch throws, SupportsPerSequenceFusedForward=true,
    /// SupportsLinearKVMigration defaults to false).
    ///
    /// Repro of the user report ("--kv-cache-dtype q4_0 makes KV cache reuse 0 on a
    /// follow-up turn; f16 reuses fully"): pre-fix, a block-quant N=1 step skipped the
    /// N=1 fast path (gated on SupportsLinearKVMigration) and fell into the
    /// ExecuteStepBatched attempt, which cleared <c>_liveCacheValid</c> BEFORE
    /// ForwardBatch threw. The per-seq fallback's EnsureOwnership then saw the
    /// stale-false flag and aborted the live-cache continuation, re-prefilling the
    /// whole conversation (PrefixCacheReusedTokens reset to 0). f16 took the fast path
    /// and never tripped that flag, hence the dtype-specific symptom.
    /// </summary>
    [Fact]
    public async Task SingleStream_BlockQuantLikeModel_LiveCacheContinuation_ReusesFullPrefix()
    {
        var model = new FusedStubModel();
        using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);

        // Turn 1: ONE sequence, prompt longer than the pooled-reuse Cap so only
        // live-cache continuation (not the capped pool) can reuse it on turn 2.
        var prompt1 = Enumerable.Repeat(1, PromptLen).ToList();
        var seq1 = new SequenceState("t1", prompt1, Round1NewTokens, BlockSize, SamplingConfig.Greedy);
        var (_, out1) = await DrainAsync(engine.SubmitRequest(seq1));

        // Turn 2: "请继续" — prompt = turn-1 (prompt + output) + a short new suffix.
        // Submitted only AFTER turn 1 fully drained, so the whole conversation runs
        // single-stream (N=1) and exercises live-cache continuation, not the
        // concurrent retained-fused path.
        var prompt2 = new List<int>(prompt1);
        prompt2.AddRange(out1);
        prompt2.AddRange(Enumerable.Repeat(PeakToken, SuffixLen));
        var seq2 = new SequenceState("t2", prompt2, 8, BlockSize, SamplingConfig.Greedy);
        var (c2, _) = await DrainAsync(engine.SubmitRequest(seq2));

        // The reused prefix must equal the entire turn-1 conversation (well past Cap):
        // only the short new suffix is re-prefilled. Pre-fix this was 0.
        Assert.True(c2.PrefixCacheReusedTokens > Cap,
            $"single-stream follow-up reused {c2.PrefixCacheReusedTokens} tokens " +
            $"(expected > {Cap}); reuse 0 is the reported q4_0 bug.");
        Assert.Equal(c2.PromptTokenCount - SuffixLen, c2.PrefixCacheReusedTokens);

        double pct = 100.0 * c2.PrefixCacheReusedTokens / c2.PromptTokenCount;
        Assert.True(pct >= 80.0, $"single-stream follow-up reuse {pct:F1}% too low");
    }

    [Fact]
    public async Task WrappedLiveCache_UnsafeRewindReprefillsWithoutTruncating()
    {
        var model = new FusedStubModel(
            supportsCrossSequenceKvReuse: false,
            supportsRetainedFusedCache: false,
            refuseWrappedRewind: true) { SupportsPrefixCheckpoints = false };
        using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);
        var prefix = Enumerable.Repeat(1, PromptLen).ToList();
        var first = await DrainAsync(engine.SubmitRequest(new SequenceState(
            "wrapped-live", prefix, 4, BlockSize, SamplingConfig.Greedy)));
        Assert.Equal(4, first.output.Count);
        var changed = new List<int>(prefix);
        changed.AddRange(first.output.Take(2));
        changed.AddRange(Enumerable.Repeat(PeakToken + 1, SuffixLen));
        var next = await DrainAsync(engine.SubmitRequest(new SequenceState(
            "wrapped-live-rewrite", changed, 4, BlockSize, SamplingConfig.Greedy)));

        Assert.Equal(SequenceStatus.FinishedLengthCapped, next.completion.Status);
        Assert.Equal(0, next.completion.PrefixCacheReusedTokens);
        Assert.Empty(model.TruncationTargets);
        Assert.Equal(Enumerable.Repeat(PeakToken, 4), next.output);
    }

    [Fact]
    public async Task SingleStream_DifferentMediaAtTheStart_DoesNotReusePlaceholderIdenticalLiveCache()
    {
        var completion = await RunSingleStreamContinuationAsync(
            firstRoundMedia: ImageAt(0, "img:a"),
            followUpMedia: ImageAt(0, "img:b"));

        Assert.Equal(0, completion.PrefixCacheReusedTokens);
    }

    [Fact]
    public async Task SingleStream_ExplicitBoundary_DoesNotReusePastClientLimit()
    {
        var completion = await RunSingleStreamContinuationAsync(
            followUpCacheBreakpoints: new[] { BlockSize });

        Assert.Equal(BlockSize, completion.PrefixCacheReusedTokens);
    }

    [Fact]
    public async Task SingleStream_ExplicitSourceBoundary_DoesNotExposeLaterLiveTokens()
    {
        var completion = await RunSingleStreamContinuationAsync(
            firstRoundCacheBreakpoints: new[] { BlockSize });

        Assert.Equal(BlockSize, completion.PrefixCacheReusedTokens);
    }

    private const int PromptLen = 24;   // > Cap so only the live holder can reuse it

    private static IReadOnlyList<PromptMediaSpan> ImageAt(int start, string contentId, int length = 8)
        => new[] { new PromptMediaSpan(start, start + length, contentId) };
    private const int Round1NewTokens = 24;
    private const int SuffixLen = 4;

    private async Task<InferenceCompletion> RunSingleStreamContinuationAsync(
        IReadOnlyList<PromptMediaSpan> firstRoundMedia = null,
        IReadOnlyList<PromptMediaSpan> followUpMedia = null,
        IReadOnlyList<int> firstRoundCacheBreakpoints = null,
        IReadOnlyList<int> followUpCacheBreakpoints = null)
    {
        var model = new FusedStubModel();
        using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);

        var prompt1 = Enumerable.Repeat(1, PromptLen).ToList();
        var seq1 = new SequenceState("single-1", prompt1, Round1NewTokens, BlockSize,
            SamplingConfig.Greedy, mediaSpans: firstRoundMedia,
            cacheBreakpoints: firstRoundCacheBreakpoints);
        var (_, out1) = await DrainAsync(engine.SubmitRequest(seq1));

        var prompt2 = new List<int>(prompt1);
        prompt2.AddRange(out1);
        prompt2.AddRange(Enumerable.Repeat(PeakToken, SuffixLen));
        var seq2 = new SequenceState("single-2", prompt2, 8, BlockSize,
            SamplingConfig.Greedy, mediaSpans: followUpMedia,
            cacheBreakpoints: followUpCacheBreakpoints);
        var (completion, _) = await DrainAsync(engine.SubmitRequest(seq2));
        return completion;
    }

    private async Task<(InferenceCompletion a, InferenceCompletion b)> RunTwoRoundsAsync(
        bool retentionEnabled,
        int? followUpCacheBoundary = null,
        int? firstRoundCacheBoundary = null,
        IReadOnlyList<PromptMediaSpan> firstRoundMedia = null,
        IReadOnlyList<PromptMediaSpan> followUpMedia = null,
        int followUpSuffixToken = PeakToken,
        Func<FusedStubModel> createModel = null,
        Microsoft.Extensions.Logging.ILogger logger = null)
    {
        string previousRetention = Environment.GetEnvironmentVariable("TS_RETAINED_FUSED_CACHE");
        string previousBudget = Environment.GetEnvironmentVariable("TS_RETAINED_FUSED_CACHE_MAX");
        string previousBatched = Environment.GetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED");
        string previousPerSeq = Environment.GetEnvironmentVariable("TS_PER_SEQ_FUSED");
        Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE", retentionEnabled ? "1" : "0");
        Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE_MAX", "4");
        Environment.SetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED", "0");
        Environment.SetEnvironmentVariable("TS_PER_SEQ_FUSED", "1");
        try
        {
            var model = createModel?.Invoke() ?? new FusedStubModel();
            using var engine = new InferenceEngine(model, Config(), logger ?? NullLogger.Instance);

            // ---- Round 1: two distinct conversations, submitted in parallel. ----
            var promptA = Enumerable.Repeat(1, PromptLen).ToList();
            var promptB = Enumerable.Repeat(2, PromptLen).ToList();
            var firstRoundBoundaries = firstRoundCacheBoundary.HasValue
                ? new List<int> { firstRoundCacheBoundary.Value }
                : null;
            var seqA1 = new SequenceState("A1", promptA, Round1NewTokens, BlockSize,
                SamplingConfig.Greedy, mediaSpans: firstRoundMedia,
                cacheBreakpoints: firstRoundBoundaries);
            var seqB1 = new SequenceState("B1", promptB, Round1NewTokens, BlockSize,
                SamplingConfig.Greedy, mediaSpans: firstRoundMedia,
                cacheBreakpoints: firstRoundBoundaries);

            // Submit BOTH before draining so the engine admits them together (N=2)
            // and serves them through the per-sequence fused path.
            var hA1 = engine.SubmitRequest(seqA1);
            var hB1 = engine.SubmitRequest(seqB1);
            var rA1 = DrainAsync(hA1);
            var rB1 = DrainAsync(hB1);
            await Task.WhenAll(rA1, rB1);
            var (_, outA1) = await rA1;
            var (_, outB1) = await rB1;

            // ---- Round 2: "请继续" — each follow-up extends its own conversation. ----
            var followA = new List<int>(promptA);
            followA.AddRange(outA1);
            followA.AddRange(Enumerable.Repeat(followUpSuffixToken, SuffixLen));
            var followB = new List<int>(promptB);
            followB.AddRange(outB1);
            followB.AddRange(Enumerable.Repeat(followUpSuffixToken, SuffixLen));

            var followUpBoundaries = followUpCacheBoundary.HasValue
                ? new List<int> { followUpCacheBoundary.Value }
                : null;
            var seqA2 = new SequenceState("A2", followA, 8, BlockSize,
                SamplingConfig.Greedy, mediaSpans: followUpMedia,
                cacheBreakpoints: followUpBoundaries);
            var seqB2 = new SequenceState("B2", followB, 8, BlockSize,
                SamplingConfig.Greedy, mediaSpans: followUpMedia,
                cacheBreakpoints: followUpBoundaries);
            var hA2 = engine.SubmitRequest(seqA2);
            var hB2 = engine.SubmitRequest(seqB2);
            var rA2 = DrainAsync(hA2);
            var rB2 = DrainAsync(hB2);
            await Task.WhenAll(rA2, rB2);
            return ((await rA2).completion, (await rB2).completion);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE", previousRetention);
            Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE_MAX", previousBudget);
            Environment.SetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED", previousBatched);
            Environment.SetEnvironmentVariable("TS_PER_SEQ_FUSED", previousPerSeq);
        }
    }

    /// <summary>
    /// The reuse a turn actually gets, and the line the log prints about it, must
    /// agree.
    ///
    /// <para>
    /// Reported 2026-09-10 against a Qwen3.8-27B server: the log carried
    /// "Live-cache continuation declined for chat-...: no live cache resident ...
    /// This turn re-prefills its full prompt (KV reuse 0)" on essentially every
    /// request, and the operator reasonably concluded the KV cache was broken. It
    /// was not - the same requests reported 73-99.9% reuse in their own completion
    /// telemetry. The live-cache attempt is simply the FIRST of three mechanisms,
    /// and on a fused model (Qwen 3.5/3.6, Gemma 4) the retained holder that runs
    /// SECOND is the one that always serves. A mechanism that has not yet let the
    /// others try cannot announce the turn's outcome.
    /// </para>
    /// </summary>
    [Fact]
    public async Task RetainedHolderServesTheTurn_LogSaysSo_AndNeverClaimsAFullRePrefill()
    {
        var log = new ReuseLogRecorder();
        var (a, b) = await RunTwoRoundsAsync(retentionEnabled: true, logger: log);

        // Precondition: this is the case the bad line used to fire on.
        Assert.Equal(a.PromptTokenCount - SuffixLen, a.PrefixCacheReusedTokens);
        Assert.Equal(b.PromptTokenCount - SuffixLen, b.PrefixCacheReusedTokens);

        // The retracted claim must be gone from every level, not just demoted.
        Assert.DoesNotContain(log.Entries, e => e.Message.Contains("re-prefills its full prompt"));
        Assert.DoesNotContain(log.Entries, e => e.Message.Contains("KV reuse 0"));

        // And the follow-ups must each be reported, truthfully, exactly once.
        foreach (string requestId in new[] { "A2", "B2" })
        {
            var reuseLines = log.Entries
                .Where(e => e.Level == Microsoft.Extensions.Logging.LogLevel.Information
                            && e.Message.Contains($"Prompt reuse for {requestId}:"))
                .ToList();
            Assert.Single(reuseLines);
            // Names the actual source: this conversation's own holder, not the public
            // shared-prefix checkpoint (the old line said "a finished request's holder or
            // a shared-prefix checkpoint", which could not tell a leak from a clone).
            Assert.Contains("a retained holder of this conversation", reuseLines[0].Message);
            Assert.DoesNotContain("shared-prefix checkpoint", reuseLines[0].Message);
            int reused = requestId == "A2" ? a.PrefixCacheReusedTokens : b.PrefixCacheReusedTokens;
            int prompt = requestId == "A2" ? a.PromptTokenCount : b.PromptTokenCount;
            Assert.Contains($"{reused}/{prompt} tokens", reuseLines[0].Message);
            Assert.Contains($"{prompt - reused} token(s) to prefill", reuseLines[0].Message);
        }
    }

    /// <summary>
    /// The other half of the contract: a turn that genuinely reuses nothing has to
    /// say so, and name every mechanism that could not help. Demoting the old line
    /// to Debug without this would have traded a false alarm for silence.
    /// </summary>
    [Fact]
    public async Task NothingRetained_LogNamesEveryMechanismThatCouldNotHelp()
    {
        var log = new ReuseLogRecorder();
        // Qwen-like: no cross-sequence snapshot reuse, so the pooled path is the one
        // that can never help - exactly the model in the 2026-09-10 report.
        var (a, b) = await RunTwoRoundsAsync(
            retentionEnabled: false,
            createModel: () => new FusedStubModel(
                supportsKvCacheTruncation: false,
                supportsCrossSequenceKvReuse: false,
                maxReusablePrefixTokens: int.MaxValue,
                supportsRetainedFusedCache: true),
            logger: log);

        Assert.Equal(0, a.PrefixCacheReusedTokens);
        Assert.Equal(0, b.PrefixCacheReusedTokens);

        var lines = log.Entries
            .Where(e => e.Message.Contains("No prompt reuse for A2:"))
            .ToList();
        Assert.Single(lines);
        string message = lines[0].Message;
        Assert.Contains("Live KV cache:", message);
        Assert.Contains("Retained state:", message);
        Assert.Contains("Pooled blocks:", message);
        // This stub model is Qwen-like: its snapshots are not cross-sequence
        // reusable, so the pooled path is the one that can never help and the
        // operator should be told that rather than left to infer it.
        Assert.Contains("unavailable for this model", message);
    }

    private sealed class ReuseLogRecorder : Microsoft.Extensions.Logging.ILogger
    {
        public List<(Microsoft.Extensions.Logging.LogLevel Level, string Message)> Entries { get; } = new();

        public IDisposable BeginScope<TState>(TState state) where TState : notnull
            => NullLogger.Instance.BeginScope(state);

        public bool IsEnabled(Microsoft.Extensions.Logging.LogLevel logLevel) => true;

        public void Log<TState>(
            Microsoft.Extensions.Logging.LogLevel logLevel,
            Microsoft.Extensions.Logging.EventId eventId,
            TState state,
            Exception exception,
            Func<TState, Exception, string> formatter)
        {
            string message = formatter != null ? formatter(state, exception) : state?.ToString() ?? string.Empty;
            lock (Entries)
                Entries.Add((logLevel, message));
        }
    }

    // ---------------------------------------------------------------------------
    // Conversation scopes: state of another conversation is reused only up to the
    // public prefix (SequenceState.SharedPrefixTokens) and never moved away from it.
    // repro-cross-session.md 2-B / 2-D, SYNTHESIS Q11 (ii).
    // ---------------------------------------------------------------------------

    private static SequenceState Scoped(
        string id, List<int> prompt, string scope, int sharedPrefix = SharedPrefixLen,
        IReadOnlyList<PromptMediaSpan> media = null, int maxNew = 4)
        => new(id, prompt, maxNew, BlockSize, SamplingConfig.Greedy,
            mediaSpans: media, sharedPrefixTokens: sharedPrefix, cacheScope: scope);

    private static List<int> Concat(params IEnumerable<int>[] parts)
    {
        var all = new List<int>();
        foreach (var part in parts) all.AddRange(part);
        return all;
    }

    /// <summary>
    /// F1/WF1: a new conversation replays conversation A's whole transcript plus its own
    /// question. It used to adopt A's retained holder (758/776 on Qwen) and MOVE it, so
    /// A's real next turn re-prefilled everything past the system prompt (536/780).
    /// </summary>
    [Fact]
    public async Task AnotherConversationsRetainedHolder_IsNotAdopted_AndItsOwnerKeepsIt()
    {
        await WithCheckpointsOnAsync(async () =>
        {
            var model = new FusedStubModel();
            using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);

            await DrainAsync(engine.SubmitRequest(Scoped("warm", Concat(SharedPrefix(), Enumerable.Repeat(5, FirstMessageLen)), "W")));
            var a1Prompt = Concat(SharedPrefix(), Enumerable.Repeat(7, FirstMessageLen));
            var a1 = await DrainAsync(engine.SubmitRequest(Scoped("A1", a1Prompt, "A")));
            Assert.Equal(SharedPrefixLen, a1.completion.PrefixCacheReusedTokens);
            // Retention runs on the engine thread just after the completion is signalled.
            Assert.True(System.Threading.SpinWait.SpinUntil(() => model.HasRetainedHolder("A1"), 5000));

            var replay = Concat(a1Prompt, a1.output, Enumerable.Repeat(9, SuffixLen));
            var b1 = await DrainAsync(engine.SubmitRequest(Scoped("B1", replay, "B")));

            Assert.Equal(SharedPrefixLen, b1.completion.PrefixCacheReusedTokens);
            Assert.True(model.HasRetainedHolder("A1"));

            var a2 = await DrainAsync(engine.SubmitRequest(Scoped("A2", replay, "A")));
            Assert.Equal(replay.Count - SuffixLen, a2.completion.PrefixCacheReusedTokens);
        });
    }

    /// <summary>
    /// R2: a different conversation whose long first message differs only in its last
    /// token used to rewind into another conversation's holder (419/426 on Gemma) and
    /// take it. A rewind into another scope's tail is not allowed at all.
    /// </summary>
    [Fact]
    public async Task AnotherConversationsHolder_IsNeverRewoundInto()
    {
        await WithCheckpointsOnAsync(async () =>
        {
            var model = new FusedStubModel();
            using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);

            await DrainAsync(engine.SubmitRequest(Scoped("warm", Concat(SharedPrefix(), Enumerable.Repeat(5, FirstMessageLen)), "W")));
            var r1Prompt = Concat(SharedPrefix(), Enumerable.Repeat(7, FirstMessageLen));
            await DrainAsync(engine.SubmitRequest(Scoped("R1", r1Prompt, "R1")));
            // Retention runs on the engine thread just after the completion is signalled.
            Assert.True(System.Threading.SpinWait.SpinUntil(() => model.HasRetainedHolder("R1"), 5000));

            var r2Prompt = Concat(SharedPrefix(), Enumerable.Repeat(7, FirstMessageLen - 1), new[] { 8 });
            var r2 = await DrainAsync(engine.SubmitRequest(Scoped("R2", r2Prompt, "R2")));

            Assert.Equal(SharedPrefixLen, r2.completion.PrefixCacheReusedTokens);
            Assert.True(model.HasRetainedHolder("R1"));
            Assert.Empty(model.TruncationTargets);
        });
    }

    [Fact]
    public async Task AnotherConversationsLiveCache_IsNotContinued_ButItsOwnNextTurnIs()
    {
        async Task<int> SecondTurnReuse(string secondScope)
        {
            var model = new FusedStubModel();
            using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);
            var prompt1 = Enumerable.Repeat(1, PromptLen).ToList();
            var first = await DrainAsync(engine.SubmitRequest(Scoped("A1", prompt1, "A", sharedPrefix: 0, maxNew: Round1NewTokens)));
            var prompt2 = Concat(prompt1, first.output, Enumerable.Repeat(PeakToken, SuffixLen));
            var second = await DrainAsync(engine.SubmitRequest(Scoped("next", prompt2, secondScope, sharedPrefix: 0)));
            return second.completion.PrefixCacheReusedTokens;
        }

        Assert.Equal(0, await SecondTurnReuse("B"));
        Assert.Equal(PromptLen + Round1NewTokens, await SecondTurnReuse("A"));
    }

    /// <summary>
    /// A3 -> B1 -> A4: conversation A ran on the primary cache; a new chat B arrived and
    /// ran on the fused path, which used to drop A's live state (A4 reused 584 of 740).
    /// A's finished state is now kept as its own retained holder when that happens.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task InterleavedNewChat_KeepsTheLiveConversationsReuse(bool scoped)
    {
        await WithCheckpointsOnAsync(async () =>
        {
            var model = new FusedStubModel();
            using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);

            var prompt = Concat(SharedPrefix(), Enumerable.Repeat(7, FirstMessageLen));
            var turn = await DrainAsync(engine.SubmitRequest(Scoped("A1", prompt, scoped ? "A" : null)));
            for (int t = 2; t <= 3; t++)
            {
                prompt = Concat(prompt, turn.output, Enumerable.Repeat(PeakToken + 1, SuffixLen));
                turn = await DrainAsync(engine.SubmitRequest(Scoped("A" + t, prompt, scoped ? "A" : null)));
                Assert.Equal(prompt.Count - SuffixLen, turn.completion.PrefixCacheReusedTokens);
            }

            var b1 = await DrainAsync(engine.SubmitRequest(Scoped("B1", Concat(SharedPrefix(), Enumerable.Repeat(8, FirstMessageLen)), scoped ? "B" : null)));
            Assert.Equal(SharedPrefixLen, b1.completion.PrefixCacheReusedTokens);

            prompt = Concat(prompt, turn.output, Enumerable.Repeat(PeakToken + 1, SuffixLen));
            var a4 = await DrainAsync(engine.SubmitRequest(Scoped("A4", prompt, scoped ? "A" : null)));
            // An unscoped caller keeps the old behaviour: no donation, the checkpoint only.
            Assert.Equal(scoped ? prompt.Count - SuffixLen : SharedPrefixLen, a4.completion.PrefixCacheReusedTokens);
        });
    }

    /// <summary>
    /// Isolation property: across random interleavings of conversations - new chats with
    /// colliding first messages, follow-up turns, replays of another conversation's whole
    /// transcript, several submitted at once - a request whose scope no earlier request
    /// had reuses at most its public prefix.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(8)]
    public async Task RandomInterleavings_ANewScopeNeverReusesPastItsPublicPrefix(int seed)
    {
        await WithCheckpointsOnAsync(async () =>
        {
            var rng = new Random(seed);
            var model = new FusedStubModel(forwardDelayMs: 0);
            using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);
            var transcripts = new List<(string Scope, List<int> Tokens)>();
            var seenScopes = new HashSet<string>(StringComparer.Ordinal);
            int nextScope = 0, nextRequest = 0, checkedNewScopes = 0;

            for (int step = 0; step < 14; step++)
            {
                int group = 1 + rng.Next(3);
                var batch = new List<(SequenceState Seq, int Transcript, bool NewScope)>();
                var busy = new HashSet<int>();
                for (int g = 0; g < group; g++)
                {
                    int action = transcripts.Count == 0 ? 0 : rng.Next(3);
                    if (action == 1)
                    {
                        int t = rng.Next(transcripts.Count);
                        if (!busy.Add(t)) continue;
                        var prompt = Concat(transcripts[t].Tokens, Enumerable.Repeat(1 + rng.Next(2), SuffixLen));
                        batch.Add((Scoped($"r{nextRequest++}", prompt, transcripts[t].Scope), t, false));
                    }
                    else
                    {
                        string scope = "s" + nextScope++;
                        List<int> prompt = action == 2
                            // A replay of someone else's transcript under a new scope.
                            ? Concat(transcripts[rng.Next(transcripts.Count)].Tokens, Enumerable.Repeat(1 + rng.Next(2), SuffixLen))
                            // A new chat; first messages collide on purpose.
                            : Concat(SharedPrefix(), Enumerable.Repeat(7 + rng.Next(2), FirstMessageLen));
                        transcripts.Add((scope, prompt));
                        batch.Add((Scoped($"r{nextRequest++}", prompt, scope), transcripts.Count - 1, true));
                    }
                }

                var handles = batch.Select(b => DrainAsync(engine.SubmitRequest(b.Seq))).ToList();
                await Task.WhenAll(handles);
                for (int i = 0; i < batch.Count; i++)
                {
                    var (seq, t, isNew) = batch[i];
                    var (completion, output) = await handles[i];
                    if (isNew && seenScopes.Add(seq.CacheScope))
                    {
                        checkedNewScopes++;
                        Assert.True(completion.PrefixCacheReusedTokens <= seq.SharedPrefixTokens,
                            $"seed {seed}: new scope {seq.CacheScope} ({seq.RequestId}) reused " +
                            $"{completion.PrefixCacheReusedTokens} > public prefix {seq.SharedPrefixTokens}");
                    }
                    seenScopes.Add(seq.CacheScope);
                    transcripts[t] = (transcripts[t].Scope, Concat(seq.PromptTokens, output));
                }
            }
            Assert.True(checkedNewScopes > 3);
        });
    }

    // ---------------------------------------------------------------------------
    // Media identity: reuse is checked positionally over the reused prefix, by
    // content. repro-image-turn.md 1-A / 1-D, SYNTHESIS Q11 (iii).
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task AnImageTurn_ContinuesTheTextBeforeTheImageFromTheLiveCache()
    {
        var model = new FusedStubModel();
        using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);
        var prompt1 = Enumerable.Repeat(1, PromptLen).ToList();
        var first = await DrainAsync(engine.SubmitRequest(Scoped("t1", prompt1, "A", sharedPrefix: 0, maxNew: Round1NewTokens)));

        var history = Concat(prompt1, first.output);
        var prompt2 = Concat(history, Enumerable.Repeat(6, 8), Enumerable.Repeat(PeakToken, SuffixLen));
        var image = ImageAt(history.Count, "img:photo");
        var second = await DrainAsync(engine.SubmitRequest(Scoped("t2-image", prompt2, "A", sharedPrefix: 0, media: image)));

        Assert.Equal(history.Count, second.completion.PrefixCacheReusedTokens);
    }

    /// <summary>
    /// Gemma 4 on a wrapped sliding window: an image prefilled after a reused prefix did
    /// not match a cold prefill, so a model that says it cannot do that gets no reuse for
    /// such a turn (and keeps it where it can, within the window).
    /// </summary>
    [Theory]
    [InlineData(1000, true)]
    [InlineData(40, false)]
    public async Task MediaAfterTheReusedPrefix_IsReusedOnlyWhereTheModelCanPrefillItExactly(int limit, bool reused)
    {
        var model = new FusedStubModel(mediaAfterReuseMaxPromptTokens: limit);
        using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);
        var prompt1 = Enumerable.Repeat(1, PromptLen).ToList();
        var first = await DrainAsync(engine.SubmitRequest(Scoped("t1", prompt1, "A", sharedPrefix: 0, maxNew: Round1NewTokens)));
        var history = Concat(prompt1, first.output);
        var prompt2 = Concat(history, Enumerable.Repeat(6, 8), Enumerable.Repeat(PeakToken, SuffixLen));
        var second = await DrainAsync(engine.SubmitRequest(Scoped("t2-image", prompt2, "A", sharedPrefix: 0,
            media: ImageAt(history.Count, "img:photo"))));

        Assert.Equal(reused ? history.Count : 0, second.completion.PrefixCacheReusedTokens);
    }

    private async Task<int> ReuseAfterAnImageTurnAsync(string sameOrOtherImage, bool modelContinuesPastMedia)
    {
        var model = new FusedStubModel(
            supportsReuseAcrossMediaSpan: modelContinuesPastMedia,
            supportsKvCacheTruncation: modelContinuesPastMedia);
        using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);
        var prompt1 = Concat(Enumerable.Repeat(1, 4), Enumerable.Repeat(6, 8), Enumerable.Repeat(1, 12));
        var first = await DrainAsync(engine.SubmitRequest(
            Scoped("t1", prompt1, "A", sharedPrefix: 0, media: ImageAt(4, "img:photo"), maxNew: Round1NewTokens)));

        var prompt2 = Concat(prompt1, first.output, Enumerable.Repeat(PeakToken, SuffixLen));
        var second = await DrainAsync(engine.SubmitRequest(
            Scoped("t2", prompt2, "A", sharedPrefix: 0, media: ImageAt(4, sameOrOtherImage))));
        return second.completion.PrefixCacheReusedTokens;
    }

    /// <summary>OpenAI clients resend the image with every turn; the same content at the
    /// same place is the same cache, and a different picture is not.</summary>
    [Fact]
    public async Task TheSameImageResent_ContinuesPastIt_AndADifferentImageDoesNot()
    {
        Assert.Equal(PromptLen + Round1NewTokens, await ReuseAfterAnImageTurnAsync("img:photo", modelContinuesPastMedia: true));
        Assert.Equal(0, await ReuseAfterAnImageTurnAsync("img:another", modelContinuesPastMedia: true));
    }

    /// <summary>Qwen 3.5 (M-RoPE, decode at absolute positions) cannot continue a cache
    /// past an image exactly, so its reuse stops at the first span.</summary>
    [Fact]
    public async Task AModelThatCannotContinuePastMedia_StopsAtTheFirstSpan()
    {
        Assert.Equal(0, await ReuseAfterAnImageTurnAsync("img:photo", modelContinuesPastMedia: false));
    }

    /// <summary>
    /// Gemma 4 turns of 512 tokens or fewer: the live cache used to be refused as "within
    /// the pooled reuse cap", and the pooled path could return at most whole blocks, so a
    /// short turn reused 0 or 256 tokens of what the live cache held.
    /// </summary>
    [Fact]
    public async Task AShortLiveCacheWithinTheSlidingWindow_IsContinued()
    {
        var model = new FusedStubModel();
        using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);
        var prompt1 = Enumerable.Repeat(1, 6).ToList();
        var first = await DrainAsync(engine.SubmitRequest(Scoped("short-1", prompt1, "A", sharedPrefix: 0, maxNew: 4)));
        var history = Concat(prompt1, first.output);
        Assert.True(history.Count < Cap);

        var prompt2 = Concat(history, Enumerable.Repeat(PeakToken + 1, SuffixLen));
        var second = await DrainAsync(engine.SubmitRequest(Scoped("short-2", prompt2, "A", sharedPrefix: 0)));

        Assert.Equal(history.Count, second.completion.PrefixCacheReusedTokens);
    }

    private static async Task<(InferenceCompletion completion, List<int> output)> DrainAsync(InferenceRequestHandle handle)
    {
        var output = new List<int>();
        await foreach (var t in handle.Tokens.ReadAllAsync())
            output.Add(t);
        var completion = await handle.Completion;
        return (completion, output);
    }

    private static bool ExecutorTracksFusedRequest(InferenceEngine engine, string requestId)
    {
        const System.Reflection.BindingFlags flags =
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        var executor = (BatchExecutor)typeof(InferenceEngine)
            .GetField("_executor", flags)!
            .GetValue(engine)!;
        var sequences = (System.Collections.IDictionary)typeof(BatchExecutor)
            .GetField("_fusedSeqById", flags)!
            .GetValue(executor)!;
        var truncations = (System.Collections.IDictionary)typeof(BatchExecutor)
            .GetField("_pendingRetainedFusedTruncations", flags)!
            .GetValue(executor)!;
        return sequences.Contains(requestId) || truncations.Contains(requestId);
    }

    private static SchedulerConfig Config() => new()
    {
        MaxNumBatchedTokens = 1024,
        MaxNumRunningSequences = 8,
        MaxPrefillChunkSize = 256,
        SoloPrefillChunkSize = 256,
        NumBlocks = 256,
        BlockSize = BlockSize,
        EnablePrefixCaching = true,
        DecodeQuantumTokens = 1, // rotate eagerly so both round-1 seqs interleave
    };

    /// <summary>
    /// Deterministic stub that mimics a sliding-window model on the per-sequence
    /// fused path: each RequestId gets its own (in-memory) K/V holder, pooled reuse
    /// is capped at <see cref="Cap"/>, and finished holders can be retained and
    /// re-keyed. Forward only tracks a per-holder token count; logits always peak at
    /// <see cref="PeakToken"/> so greedy decode is deterministic.
    /// </summary>
    private sealed class FusedStubModel : IModelArchitecture, IBatchedPagedModel, ISpeculativeTarget, IExactFusedCacheReuse
    {
        private sealed class Holder { public int SeqLen; }

        // ---- ISpeculativeTarget: a verify over the active holder, rows all peaking
        // at PeakToken, so a lookup drafter's proposals are accepted and the stream
        // stays what plain decoding produces.
        public int SpecForwardCalls { get; private set; }
        public int CacheSeqLen => Active.SeqLen;
        public int MaxContextLength => 4096;
        public bool SpeculationProfitable => true;
        public void SpecForward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            SpecForwardCalls++;
            int startPos = Active.SeqLen;
            Active.SeqLen += tokens.Length;
            int rows = allLogitsRows ? tokens.Length : 1;
            Array.Clear(logitsOut, 0, rows * VocabSize);
            for (int r = 0; r < rows; r++)
            {
                // Row r predicts the token after position startPos + r, exactly as
                // Forward does for a cache that then holds startPos + r + 1 tokens.
                int position = allLogitsRows ? startPos + r + 1 : Active.SeqLen;
                logitsOut[r * VocabSize + PeakAt(position)] = 10.0f;
            }
        }
        public bool SpecTrunkFollowsBoundCache => true;
        public void SpecEnsureCapacity(int requiredSeqLen) { }
        public void SpecSnapshotRecurrentState() { }
        public void SpecRestoreRecurrentState() { }
        public void SpecRewindCache(int length) => Active.SeqLen = length;

        private readonly Dictionary<string, Holder> _holders = new(StringComparer.Ordinal);
        private readonly Dictionary<string, Holder> _retained = new(StringComparer.Ordinal);
        /// <summary>Checkpoint key -> the token count the active cache held when it was copied.</summary>
        public Dictionary<string, int> Checkpoints { get; } = new(StringComparer.Ordinal);
        public int Clones { get; private set; }
        public bool SupportsPrefixCheckpoints { get; set; } = true;
        private readonly List<string> _releasedRequestIds = new();
        private readonly List<string> _discardedRetainedRequestIds = new();
        private readonly object _lifecycleLock = new();
        private readonly int _forwardDelayMs;
        private readonly bool _supportsKvCacheTruncation;
        private readonly bool _supportsCrossSequenceKvReuse;
        private readonly int _maxReusablePrefixTokens;
        private readonly bool _supportsRetainedFusedCache;
        private readonly bool _batchedFusedDecodeSucceeds;
        private readonly bool _refuseWrappedRewind;
        private readonly bool _refuseTruncationAtExecution;
        private string _activeKey;            // null => primary active
        private Holder _primary = new();

        private Holder Active => _activeKey == null ? _primary : _holders[_activeKey];

        // With periodicPeak the argmax alternates PeakToken / PeakToken+1 by cache
        // position, so the stream repeats with period two and a lookup drafter has
        // something to find (a constant stream never has a token AFTER its last
        // occurrence to propose).
        private readonly bool _periodicPeak;
        private int PeakAt(int position) => _periodicPeak ? PeakToken + (position % 2) : PeakToken;

        public FusedStubModel(
            bool peakIsEos = false,
            int forwardDelayMs = 0,
            bool supportsKvCacheTruncation = true,
            bool supportsCrossSequenceKvReuse = true,
            int maxReusablePrefixTokens = Cap,
            bool supportsRetainedFusedCache = true,
            bool batchedFusedDecodeSucceeds = false,
            bool periodicPeak = false,
            bool refuseWrappedRewind = false,
            bool refuseTruncationAtExecution = false,
            bool supportsReuseAcrossMediaSpan = true,
            int mediaAfterReuseMaxPromptTokens = int.MaxValue)
        {
            SupportsReuseAcrossMediaSpan = supportsReuseAcrossMediaSpan;
            _mediaAfterReuseMaxPromptTokens = mediaAfterReuseMaxPromptTokens;
            _refuseWrappedRewind = refuseWrappedRewind;
            _refuseTruncationAtExecution = refuseTruncationAtExecution;
            _periodicPeak = periodicPeak;
            Tokenizer = new StubTokenizer(peakIsEos);
            _forwardDelayMs = forwardDelayMs;
            _supportsKvCacheTruncation = supportsKvCacheTruncation;
            _supportsCrossSequenceKvReuse = supportsCrossSequenceKvReuse;
            _maxReusablePrefixTokens = maxReusablePrefixTokens;
            _supportsRetainedFusedCache = supportsRetainedFusedCache;
            _batchedFusedDecodeSucceeds = batchedFusedDecodeSucceeds;
        }

        public IReadOnlyList<string> DiscardedRetainedRequestIds
        {
            get
            {
                lock (_lifecycleLock)
                    return _discardedRetainedRequestIds.ToArray();
            }
        }

        public bool WasReleased(string requestId)
        {
            lock (_lifecycleLock)
                return _releasedRequestIds.Contains(requestId);
        }

        public ModelConfig Config { get; } = new ModelConfig { VocabSize = VocabSize };
        public ITokenizer Tokenizer { get; }
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => _supportsKvCacheTruncation;
        public bool CanTruncateKVCache(int cachedTokenCount, int targetTokenCount)
            => targetTokenCount >= 0 && targetTokenCount <= cachedTokenCount
                && (targetTokenCount == cachedTokenCount
                    || (_supportsKvCacheTruncation
                        && (!_refuseWrappedRewind || cachedTokenCount <= Cap || targetTokenCount == 0)));
        public bool SupportsExactFusedCacheReuse { get; set; }
        public int KVCacheTruncationGranularity { get; set; } = 1;
        public int ExactQueryHeadOffset { get; set; }
        public List<(bool Retained, int Cached, int Target)> ExactQueries { get; } = new();
        public bool CanReuseLivePrefix(int cachedTokenCount, int targetTokenCount)
        {
            ExactQueries.Add((false, cachedTokenCount, targetTokenCount));
            return Active.SeqLen + ExactQueryHeadOffset == cachedTokenCount
                && CanTruncateKVCache(cachedTokenCount, targetTokenCount)
                && (targetTokenCount == cachedTokenCount || targetTokenCount % KVCacheTruncationGranularity == 0);
        }
        public bool CanReuseRetainedPrefix(string key, int cachedTokenCount, int targetTokenCount)
        {
            ExactQueries.Add((true, cachedTokenCount, targetTokenCount));
            return _retained.TryGetValue(key, out var holder) && holder.SeqLen + ExactQueryHeadOffset == cachedTokenCount
                && CanTruncateKVCache(cachedTokenCount, targetTokenCount)
                && (targetTokenCount == cachedTokenCount || targetTokenCount % KVCacheTruncationGranularity == 0);
        }
        public bool HasRetainedHolder(string requestId) => _retained.ContainsKey(requestId);
        public int RetainCalls { get; private set; }
        public int RebindCalls { get; private set; }
        public string ThrowOnDiscardKey { get; set; }
        public int ExecutionTruncationRefusals { get; private set; }
        public List<(string RequestId, int Start, int Count)> ForwardCalls { get; } = new();
        public List<int> TruncationTargets { get; } = new();
        public int SuccessfulBatchedFusedDecodeCalls { get; private set; }

        // The fused path never reads paged storage, but the engine still sizes the
        // block pool from this, so it must be > 0.
        public long ComputeKVBlockByteSize(int tokenCount) => 32L * tokenCount;

        public float[] Forward(int[] tokens)
        {
            if (_forwardDelayMs > 0)
                System.Threading.Thread.Sleep(_forwardDelayMs);
            ForwardCalls.Add((_activeKey, Active.SeqLen, tokens.Length));
            Active.SeqLen += tokens.Length;
            var logits = new float[VocabSize];
            logits[PeakAt(Active.SeqLen)] = 10.0f;
            return logits;
        }

        public void ResetKVCache() => Active.SeqLen = 0;
        public bool TryTruncateKVCache(int tokenCount)
        {
            if (_refuseTruncationAtExecution && tokenCount < Active.SeqLen)
            {
                ExecutionTruncationRefusals++;
                return false;
            }
            TruncateKVCache(tokenCount);
            return true;
        }
        public void TruncateKVCache(int tokenCount)
        {
            if (!_supportsKvCacheTruncation)
                throw new InvalidOperationException("non-truncatable fused holder was truncated");
            TruncationTargets.Add(tokenCount);
            if (!CanTruncateKVCache(Active.SeqLen, tokenCount))
                throw new InvalidOperationException("unsafe wrapped fused holder was truncated");
            Active.SeqLen = Math.Min(Active.SeqLen, tokenCount);
        }
        public void Dispose() { }

        // Snapshot/cross-request block reuse and retained-holder reuse are
        // deliberately configurable independently.
        public bool SupportsKVStateSnapshot => true;
        public bool SupportsCrossSequenceKvReuse => _supportsCrossSequenceKvReuse;
        public int MaxReusablePrefixTokens => _maxReusablePrefixTokens;
        public bool SupportsReuseAcrossMediaSpan { get; }
        private readonly int _mediaAfterReuseMaxPromptTokens;
        public bool CanPrefillMediaAfterReusedPrefix(int promptTokens) => promptTokens <= _mediaAfterReuseMaxPromptTokens;
        public string KVStateFingerprint => "fused-stub";
        public bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            destination.Clear();
            return true;
        }
        public bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source) => true;

        // ---- IBatchedPagedModel: per-sequence fused forward + retention ----
        public bool SupportsPerSequenceFusedForward => true;
        public bool SupportsRetainedFusedCache => _supportsRetainedFusedCache;

        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
            => throw new NotSupportedException("fused stub only serves the per-sequence fused path");

        public bool BindSequenceCache(string requestId)
        {
            if (string.Equals(_activeKey, requestId, StringComparison.Ordinal)) return false;
            bool fresh;
            if (!_holders.TryGetValue(requestId, out var h)) { h = new Holder(); _holders[requestId] = h; fresh = true; }
            else fresh = false;
            _activeKey = requestId;
            return fresh;
        }

        public void AdoptPrimaryCacheToFused(string requestId)
        {
            if (_activeKey != null || _holders.ContainsKey(requestId)) return;
            _holders[requestId] = _primary; // hand the live primary state to the holder
            _activeKey = requestId;
            _primary = new Holder();
        }

        public void RestorePrimaryCache()
        {
            // Holders are referenced objects in _holders, so just repoint to primary.
            if (_activeKey != null) _activeKey = null;
        }

        public bool HasFusedSequenceCache(string requestId) => _holders.ContainsKey(requestId);

        public bool CanBatchDecode(string requestId, int position)
            => _batchedFusedDecodeSucceeds && _holders.ContainsKey(requestId);

        public bool TryForwardBatchedFusedDecode(
            IReadOnlyList<string> requestIds, int[] tokens, int[] positions, float[][] outLogits)
        {
            if (!_batchedFusedDecodeSucceeds || requestIds.Count != tokens.Length
                || requestIds.Count != positions.Length || requestIds.Count != outLogits.Length)
            {
                return false;
            }

            for (int i = 0; i < requestIds.Count; i++)
            {
                if (!_holders.TryGetValue(requestIds[i], out var holder))
                    return false;
                holder.SeqLen = positions[i] + 1;
                var logits = new float[VocabSize];
                logits[PeakAt(holder.SeqLen)] = 10.0f;
                outLogits[i] = logits;
            }
            SuccessfulBatchedFusedDecodeCalls++;
            return true;
        }

        public void OnSequenceReleased(string requestId)
        {
            lock (_lifecycleLock)
                _releasedRequestIds.Add(requestId);
            if (string.Equals(_activeKey, requestId, StringComparison.Ordinal)) _activeKey = null;
            _holders.Remove(requestId);
        }

        public bool RetainSequenceCache(string requestId)
        {
            RetainCalls++;
            if (!_holders.TryGetValue(requestId, out var h)) return false;
            if (string.Equals(_activeKey, requestId, StringComparison.Ordinal)) _activeKey = null;
            _holders.Remove(requestId);
            _retained[requestId] = h;
            return true;
        }

        public bool TryRebindRetainedCache(string retainedRequestId, string newRequestId)
        {
            RebindCalls++;
            if (!_retained.TryGetValue(retainedRequestId, out var h)) return false;
            _retained.Remove(retainedRequestId);
            _holders[newRequestId] = h;
            return true;
        }

        public void DiscardRetainedCache(string requestId)
        {
            if (requestId == ThrowOnDiscardKey)
                throw new InvalidOperationException("controlled native discard refusal");
            if (!_retained.Remove(requestId)) return;
            lock (_lifecycleLock)
                _discardedRetainedRequestIds.Add(requestId);
        }

        public bool TryCheckpointActiveCache(string key)
        {
            if (_retained.ContainsKey(key) || _holders.ContainsKey(key)) return false;
            _retained[key] = new Holder { SeqLen = Active.SeqLen };   // an independent copy
            Checkpoints[key] = Active.SeqLen;
            return true;
        }

        public bool TryCloneRetainedCache(string retainedKey, string newRequestId)
        {
            if (!_retained.TryGetValue(retainedKey, out var h)) return false;
            if (_holders.ContainsKey(newRequestId)) return false;
            _holders[newRequestId] = new Holder { SeqLen = h.SeqLen };
            Clones++;
            return true;
        }

        // A checkpoint on disk is the holder's token count, which is all the state
        // this stub has; a real model writes its K/V and recurrent state the same way.
        public int Exports { get; private set; }
        public int Imports { get; private set; }
        public bool SupportsRetainedCacheSerialization { get; set; } = true;

        public bool TryExportRetainedCache(string key, System.IO.Stream destination)
        {
            if (!_retained.TryGetValue(key, out var h)) return false;
            var w = new System.IO.BinaryWriter(destination, System.Text.Encoding.UTF8, leaveOpen: true);
            w.Write(0x53545542u);   // "STUB"
            w.Write(h.SeqLen);
            w.Flush();
            Exports++;
            return true;
        }

        public bool TryImportRetainedCache(string key, System.IO.Stream source)
        {
            if (_retained.ContainsKey(key) || _holders.ContainsKey(key)) return false;
            var r = new System.IO.BinaryReader(source, System.Text.Encoding.UTF8, leaveOpen: true);
            if (r.ReadUInt32() != 0x53545542u) return false;
            int seqLen = r.ReadInt32();
            if (seqLen <= 0) return false;
            _retained[key] = new Holder { SeqLen = seqLen };
            Imports++;
            return true;
        }

        private sealed class StubTokenizer : ITokenizer
        {
            private readonly bool _peakIsEos;

            public StubTokenizer(bool peakIsEos)
            {
                _peakIsEos = peakIsEos;
                Vocab = new string[RetainedFusedCacheTests.VocabSize];
                for (int i = 0; i < RetainedFusedCacheTests.VocabSize; i++) Vocab[i] = i.ToString();
            }
            public string[] Vocab { get; }
            public int BosTokenId => -1;
            public int[] EosTokenIds => _peakIsEos ? new[] { PeakToken } : Array.Empty<int>();
            public int VocabSize => Vocab.Length;
            public List<int> Encode(string text, bool addSpecial = true) => new();
            public string Decode(List<int> ids) => string.Join(",", ids);
            public void AppendTokenBytes(int tokenId, List<byte> buffer)
            {
                foreach (var b in System.Text.Encoding.UTF8.GetBytes(tokenId.ToString())) buffer.Add(b);
            }
            public bool IsEos(int tokenId) => _peakIsEos && tokenId == PeakToken;
            public int LookupToken(string tokenStr) => -1;
        }
    }
}
