// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// A waiting request's prompt block hashes are computed once, not on every
/// Schedule() call its capacity check fails. Caching them must not change a single
/// admission decision, however the pool changes between calls.
/// </summary>
public class PromptBlockHashCacheTests
{
    private readonly ITestOutputHelper _output;

    public PromptBlockHashCacheTests(ITestOutputHelper output) { _output = output; }

    /// <summary>
    /// Two schedulers, one caching and one hashing every plan, driven through the same
    /// random workload: shared prefixes, pictures, conversation scopes, breakpoints,
    /// staggered arrivals and a pool small enough that requests wait for capacity, adopt
    /// pooled blocks, preempt and re-adopt, and evict cached blocks. Every step's plan,
    /// every preemption, every block id and every reuse count must be identical.
    /// </summary>
    [Theory]
    [InlineData(20260917, false)]
    [InlineData(918273, false)]
    [InlineData(5550123, true)]
    public void CachedHashes_ProduceIdenticalSchedules_AcrossPoolChanges(int seed, bool perBlockCapture)
    {
        const int blockSize = 8;
        var rng = new Random(seed);
        int cachedComputations = 0, uncachedComputations = 0, sequences = 0, waitingSteps = 0, adoptedTokens = 0;
        for (int trial = 0; trial < 60; trial++)
        {
            int numBlocks = rng.Next(10, 28);
            var cfg = new SchedulerConfig
            {
                MaxNumBatchedTokens = rng.Next(16, 96),
                MaxNumRunningSequences = rng.Next(2, 5),
                MaxPrefillChunkSize = rng.Next(4, 32),
                SoloPrefillChunkSize = 64,
                NumBlocks = numBlocks,
                BlockSize = blockSize,
                EnablePrefixCaching = true,
                DecodeQuantumTokens = 1,
                StopRepetition = false,
            };
            var poolCached = new BlockPool(numBlocks, blockSize, 0);
            var poolUncached = new BlockPool(numBlocks, blockSize, 0);
            var cached = new ContinuousBatchScheduler(cfg, poolCached, "fp-hash-cache", NullLogger.Instance,
                requiresPerBlockCapture: perBlockCapture);
            var uncached = new ContinuousBatchScheduler(cfg, poolUncached, "fp-hash-cache", NullLogger.Instance,
                requiresPerBlockCapture: perBlockCapture)
            { CachePromptBlockHashes = false };

            var prefixes = Enumerable.Range(0, 3)
                .Select(p => Enumerable.Range(0, rng.Next(2, 6) * blockSize).Select(i => 10 + p * 100 + i).ToArray())
                .ToArray();
            int count = rng.Next(3, 8);
            var arrivals = new List<(int Step, Func<SequenceState> Make)>();
            for (int i = 0; i < count; i++)
            {
                int[] prefix = prefixes[rng.Next(prefixes.Length)];
                int[] tail = Enumerable.Range(0, rng.Next(1, 3 * blockSize)).Select(_ => rng.Next(1000, 1100)).ToArray();
                int[] prompt = prefix.Concat(tail).ToArray();
                int capacity = numBlocks * blockSize;
                if (prompt.Length + 2 > capacity) prompt = prompt.Take(capacity / 2).ToArray();
                int maxNew = rng.Next(1, Math.Max(2, Math.Min(12, capacity - prompt.Length)));
                string scope = rng.Next(3) switch { 0 => null, 1 => "chat-a", _ => "chat-b" };
                int shared = scope == null ? 0 : Math.Min(prefix.Length, rng.Next(0, 3) * blockSize);
                PromptMediaSpan[] media = null;
                if (prompt.Length > 3 * blockSize && rng.Next(3) == 0)
                {
                    int start = rng.Next(blockSize, prompt.Length - blockSize);
                    media = new[] { new PromptMediaSpan(start, start + rng.Next(1, blockSize), rng.Next(2) == 0 ? "picture-1" : "picture-2") };
                }
                int[] breakpoints = rng.Next(6) == 0 ? new[] { rng.Next(0, prompt.Length) } : null;
                string id = $"t{trial}-s{i}";
                arrivals.Add((rng.Next(0, 25), () => new SequenceState(id, prompt.ToList(), maxNew, blockSize, SamplingConfig.Greedy,
                    mediaSpans: media, cacheBreakpoints: breakpoints, sharedPrefixTokens: shared, cacheScope: scope)));
            }
            arrivals.Sort((a, b) => a.Step.CompareTo(b.Step));
            var seqCached = arrivals.Select(a => a.Make()).ToList();
            var seqUncached = arrivals.Select(a => a.Make()).ToList();
            sequences += count;

            int next = 0;
            for (int step = 0; step < 5000 && seqCached.Any(s => !s.Status.IsFinished()); step++)
            {
                while (next < arrivals.Count && arrivals[next].Step <= step)
                {
                    cached.Submit(seqCached[next]);
                    uncached.Submit(seqUncached[next]);
                    next++;
                }
                var outCached = cached.Schedule();
                var outUncached = uncached.Schedule();
                if (cached.WaitingCount > 0) waitingSteps++;

                string where = $"seed {seed} trial {trial} step {step}";
                Assert.Equal(Describe(outUncached), Describe(outCached));
                for (int i = 0; i < seqCached.Count; i++)
                {
                    var c = seqCached[i];
                    var u = seqUncached[i];
                    Assert.True(u.Status == c.Status, $"{where}: {c.RequestId} status {c.Status} vs {u.Status}");
                    Assert.True(u.PrefixCacheReusedTokens == c.PrefixCacheReusedTokens,
                        $"{where}: {c.RequestId} reused {c.PrefixCacheReusedTokens} vs {u.PrefixCacheReusedTokens}");
                    Assert.Equal(u.BlockTable.Blocks.Select(b => b.Id), c.BlockTable.Blocks.Select(b => b.Id));
                }

                Apply(cached, outCached);
                Apply(uncached, outUncached);
            }
            Assert.All(seqCached, s => Assert.True(s.Status.IsFinished(), $"seed {seed} trial {trial}: {s} did not finish"));
            Assert.Equal(poolUncached.NumFreeBlocks, poolCached.NumFreeBlocks);
            cachedComputations += cached.PromptBlockHashComputations;
            uncachedComputations += uncached.PromptBlockHashComputations;
            adoptedTokens += seqCached.Sum(s => s.PrefixCacheReusedTokens);
        }

        _output.WriteLine($"seed {seed}: {sequences} sequences, {waitingSteps} steps with a waiting request, " +
                          $"{adoptedTokens} tokens adopted; prompt hashing {cachedComputations} cached vs {uncachedComputations} uncached");
        // The workload does exercise waiting, adoption and repeated planning.
        Assert.True(waitingSteps > 0);
        Assert.True(adoptedTokens > 0);
        Assert.True(uncachedComputations > cachedComputations);
        Assert.True(cachedComputations <= sequences, $"{cachedComputations} prompt hashings for {sequences} sequences");
    }

    private static string Describe(SchedulerOutput output)
        => string.Join(";", output.ScheduledWork.Select(w => $"{w.Sequence.RequestId}:{w.NumScheduledTokens}:{w.IsPrefill}"))
           + "|" + string.Join(",", output.PreemptedRequestIds);

    private static void Apply(ContinuousBatchScheduler sched, SchedulerOutput output)
    {
        foreach (var work in output.ScheduledWork)
        {
            var seq = work.Sequence;
            int before = seq.NumComputedTokens;
            seq.AdvanceComputedTokens(work.NumScheduledTokens);
            // Every committed block holds K/V a later request could read, as a capture would leave it.
            for (int b = before / sched.Config.BlockSize; b < seq.NumComputedTokens / sched.Config.BlockSize; b++)
            {
                var block = seq.BlockTable.Blocks[b];
                block.Used = sched.Config.BlockSize;
                block.IsRestorablePrefixEnd = true;
            }
            sched.OnBlocksCommitted(seq, before);
            if (seq.NumComputedTokens < seq.NumTotalTokens) continue;
            seq.AppendOutputToken(3);
            if (seq.ShouldStopForLength())
                sched.NotifyStop(seq, SequenceStatus.FinishedLengthCapped, "length", output);
        }
    }

    /// <summary>The cache is keyed on the scheduler's fingerprint and block size: a
    /// sequence planned by one engine and then by another (a rebuilt engine after a model
    /// change) gets that engine's hashes, exactly as if it had never been planned.</summary>
    [Fact]
    public void CachedHashes_AreRecomputed_ForAnotherFingerprintOrBlockSize()
    {
        var prompt = Enumerable.Range(1, 40).ToList();
        var seq = new SequenceState("s", prompt, 4, 8, SamplingConfig.Greedy,
            mediaSpans: new[] { new PromptMediaSpan(12, 20, "picture") }, sharedPrefixTokens: 8, cacheScope: "chat");

        var hashes = new List<(ContinuousBatchScheduler Sched, int BlockSize, string Fingerprint)>();
        foreach (var (fp, bs) in new[] { ("fp-one", 8), ("fp-two", 8), ("fp-two", 16), ("fp-one", 8) })
        {
            var pool = new BlockPool(16, bs, 0);
            var sched = new ContinuousBatchScheduler(
                new SchedulerConfig { NumBlocks = 16, BlockSize = bs, EnablePrefixCaching = true, MaxNumBatchedTokens = 64 },
                pool, fp, NullLogger.Instance);
            var got = sched.GetPromptBlockHashesForTest(seq);
            var fresh = new SequenceState("fresh", prompt, 4, bs, SamplingConfig.Greedy,
                mediaSpans: seq.MediaSpans, sharedPrefixTokens: 8, cacheScope: "chat");
            var reference = new ContinuousBatchScheduler(
                new SchedulerConfig { NumBlocks = 16, BlockSize = bs, EnablePrefixCaching = true, MaxNumBatchedTokens = 64 },
                new BlockPool(16, bs, 0), fp, NullLogger.Instance) { CachePromptBlockHashes = false };
            Assert.Equal(reference.GetPromptBlockHashesForTest(fresh), got);
            Assert.Equal(1, sched.PromptBlockHashComputations);
            // A second plan on the same scheduler is served from the cache.
            Assert.Equal(got, sched.GetPromptBlockHashesForTest(seq));
            Assert.Equal(1, sched.PromptBlockHashComputations);
        }
    }

    /// <summary>
    /// Microbenchmark: a queued 20k-token prompt whose capacity check fails on every
    /// Schedule() call (a longer request is still prefilling and owns most of the pool,
    /// and the prefix they share does not cover the difference). Reports allocation and
    /// latency per call with and without the cache; the cached plan must not allocate
    /// per prompt block.
    /// </summary>
    [Fact]
    public void QueuedLongPrompt_ScheduleCost_WithAndWithoutCachedHashes()
    {
        var withCache = MeasureWaitingSchedule(cacheHashes: true);
        var without = MeasureWaitingSchedule(cacheHashes: false);
        _output.WriteLine($"queued 20000-token prompt, {without.PromptBlocks} full blocks, per Schedule() call:");
        _output.WriteLine($"  hashes recomputed: {without.BytesPerCall,8:N0} bytes, {without.MicrosPerCall,8:F1} us ({without.Computations} prompt hashings in {without.Calls} calls)");
        _output.WriteLine($"  hashes cached:     {withCache.BytesPerCall,8:N0} bytes, {withCache.MicrosPerCall,8:F1} us ({withCache.Computations} prompt hashings in the whole run)");

        // Once for "a" at its admission, once for "b" however often it is planned.
        Assert.Equal(2, withCache.Computations);
        Assert.Equal(without.Calls, without.Computations);
        // A Schedule() call allocates its output and snapshots either way; what the cache
        // removes is the per-block hashing (an IncrementalHash and digest per block).
        Assert.True(withCache.BytesPerCall * 4 < without.BytesPerCall,
            $"cached {withCache.BytesPerCall} bytes/call vs uncached {without.BytesPerCall}");
        Assert.True(withCache.BytesPerCall < without.PromptBlocks * 16,
            $"cached {withCache.BytesPerCall} bytes/call still scales with the {without.PromptBlocks}-block prompt");
    }

    private sealed record WaitingCost(int Calls, int Computations, int PromptBlocks, double BytesPerCall, double MicrosPerCall);

    private static WaitingCost MeasureWaitingSchedule(bool cacheHashes)
    {
        const int blockSize = 256;
        const int numBlocks = 200;
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 16384,
            MaxNumRunningSequences = 4,
            MaxPrefillChunkSize = 4096,
            SoloPrefillChunkSize = 16384,
            NumBlocks = numBlocks,
            BlockSize = blockSize,
            EnablePrefixCaching = true,
            DecodeQuantumTokens = 1,
            StopRepetition = false,
        };
        var pool = new BlockPool(numBlocks, blockSize, 0);
        var sched = new ContinuousBatchScheduler(cfg, pool, "fp-hash-cache-bench", NullLogger.Instance)
        { CachePromptBlockHashes = cacheHashes };

        // "a": 4096-token shared prefix + 44000 more (188 blocks), mid-prefill. "b": the
        // same prefix + 15904 tokens (79 blocks, 16 of them shared), with a picture and a
        // conversation scope so every block carries its salts.
        var prefix = Enumerable.Range(1, 4096).ToList();
        var a = new SequenceState("a", prefix.Concat(Enumerable.Range(100000, 44000)).ToList(), 16, blockSize, SamplingConfig.Greedy);
        var b = new SequenceState("b", prefix.Concat(Enumerable.Range(200000, 15904)).ToList(), 16, blockSize, SamplingConfig.Greedy,
            mediaSpans: new[] { new PromptMediaSpan(6000, 6600, "picture-content-id") },
            sharedPrefixTokens: 4096, cacheScope: "conversation-b");

        sched.Submit(a);
        var first = sched.Schedule();
        foreach (var work in first.ScheduledWork)
        {
            int before = work.Sequence.NumComputedTokens;
            work.Sequence.AdvanceComputedTokens(work.NumScheduledTokens);
            sched.OnBlocksCommitted(work.Sequence, before);
        }
        Assert.Equal(16384, a.NumComputedTokens);
        sched.Submit(b);

        // Warm up (JIT, first hashing), then measure the steady state of a waiting request.
        for (int i = 0; i < 20; i++)
        {
            sched.Schedule();
            Assert.Equal(SequenceStatus.Waiting, b.Status);
        }
        int computationsBefore = sched.PromptBlockHashComputations;
        const int calls = 200;
        GC.Collect();
        long bytes = GC.GetAllocatedBytesForCurrentThread();
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < calls; i++)
            sched.Schedule();
        sw.Stop();
        bytes = GC.GetAllocatedBytesForCurrentThread() - bytes;
        Assert.Equal(SequenceStatus.Waiting, b.Status);
        return new WaitingCost(
            calls,
            cacheHashes ? sched.PromptBlockHashComputations : sched.PromptBlockHashComputations - computationsBefore,
            b.PromptTokens.Count / blockSize,
            (double)bytes / calls,
            sw.Elapsed.TotalMilliseconds * 1000.0 / calls);
    }
}
