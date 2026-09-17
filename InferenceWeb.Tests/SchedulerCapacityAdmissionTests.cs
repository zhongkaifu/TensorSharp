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
/// Admission by KV-pool capacity and a preemption order that never lets a newer
/// sequence take an older one's blocks. Reproduces the Qwen 3.8 campaign case in
/// miniature: four long prompts against a pool that holds three of them used to
/// be admitted on their first chunk, prefill in parallel until the pool ran dry
/// and then preempt each other's nearly finished prefills (66 preemptions, one
/// request re-prefilled 15 times, waves of 700 s against 39 s solo).
/// </summary>
public class SchedulerCapacityAdmissionTests
{
    private const int BlockSize = 8;

    private static SchedulerConfig Config(int numBlocks) => new()
    {
        MaxNumBatchedTokens = 64,
        MaxNumRunningSequences = 4,
        MaxPrefillChunkSize = 16,
        SoloPrefillChunkSize = 64,
        NumBlocks = numBlocks,
        BlockSize = BlockSize,
        EnablePrefixCaching = false,
        DecodeQuantumTokens = 1,
        StopRepetition = false,
    };

    private static SequenceState Sequence(string id, int prompt, int maxNew)
        => new(id, Enumerable.Range(1, prompt).ToList(), maxNew, BlockSize, SamplingConfig.Greedy);

    private sealed record DriveStats(
        int Steps, int Preemptions, int PrefillTokensForwarded, int MaxRunning, List<string> FinishOrder);

    /// <summary>Plays the executor: forwards each scheduled token count, samples a
    /// token once a sequence has caught up with its whole token list, and finishes
    /// it at its length cap.</summary>
    private static DriveStats Drive(ContinuousBatchScheduler sched, IReadOnlyCollection<SequenceState> all, int maxSteps)
    {
        int steps = 0, preemptions = 0, prefill = 0, maxRunning = 0;
        var finished = new List<string>();
        while (steps < maxSteps && all.Any(s => !s.Status.IsFinished()))
        {
            var output = sched.Schedule();
            steps++;
            preemptions += output.PreemptedRequestIds.Count;
            maxRunning = Math.Max(maxRunning, sched.RunningCount);
            Assert.False(output.ScheduledWork.Count == 0 && sched.RunningCount > 0 && output.PreemptedRequestIds.Count == 0,
                "the scheduler returned an empty plan with running sequences (the engine would fail them as stalled)");
            foreach (var work in output.ScheduledWork)
            {
                var seq = work.Sequence;
                if (work.IsPrefill) prefill += work.NumScheduledTokens;
                seq.AdvanceComputedTokens(work.NumScheduledTokens);
                if (seq.NumComputedTokens < seq.NumTotalTokens)
                    continue;
                seq.AppendOutputToken(3);
                if (seq.ShouldStopForLength())
                {
                    sched.NotifyStop(seq, SequenceStatus.FinishedLengthCapped, "length", output);
                    finished.Add(seq.RequestId);
                }
            }
        }
        return new DriveStats(steps, preemptions, prefill, maxRunning, finished);
    }

    [Fact]
    public void LongPromptsBeyondPoolCapacity_AreAdmittedByCapacity_AndPrefillExactlyOnce()
    {
        // 32 blocks x 8 = 256 token slots. Each request needs 80 prompt + 8 output =
        // 88 slots (11 blocks): three fit, the fourth must wait for one to finish.
        var cfg = Config(numBlocks: 32);
        var pool = new BlockPool(cfg.NumBlocks, cfg.BlockSize, 0);
        var sched = new ContinuousBatchScheduler(cfg, pool, "fp-capacity", NullLogger.Instance);
        var seqs = Enumerable.Range(0, 4).Select(i => Sequence($"long-{i}", 80, 8)).ToArray();
        foreach (var s in seqs) sched.Submit(s);

        var stats = Drive(sched, seqs, maxSteps: 2000);

        Assert.All(seqs, s => Assert.Equal(SequenceStatus.FinishedLengthCapped, s.Status));
        Assert.All(seqs, s => Assert.Equal(8, s.OutputTokens.Count));
        Assert.Equal(0, stats.Preemptions);
        Assert.Equal(4 * 80, stats.PrefillTokensForwarded);
        Assert.Equal(3, stats.MaxRunning);
        // The waiting request runs after the first three, not in place of one of them.
        Assert.Equal("long-3", stats.FinishOrder[^1]);
        Assert.Equal(cfg.NumBlocks, pool.NumFreeBlocks);
    }

    [Fact]
    public void WaitingRequest_IsHeldUntilItsWholePromptFits()
    {
        var cfg = Config(numBlocks: 32);
        var pool = new BlockPool(cfg.NumBlocks, cfg.BlockSize, 0);
        var sched = new ContinuousBatchScheduler(cfg, pool, "fp-capacity-hold", NullLogger.Instance);
        var seqs = Enumerable.Range(0, 4).Select(i => Sequence($"long-{i}", 80, 8)).ToArray();
        foreach (var s in seqs) sched.Submit(s);

        var first = sched.Schedule();

        // Only the first chunks were needed to start all four, which is exactly what
        // used to happen; the fourth prompt's 10 blocks are not available on top of
        // the 30 the first three still need.
        Assert.Equal(3, first.ScheduledWork.Count);
        Assert.Equal(3, sched.RunningCount);
        Assert.Equal(1, sched.WaitingCount);
        Assert.Equal(SequenceStatus.Waiting, seqs[3].Status);
        Assert.Equal(0, seqs[3].BlockTable.NumBlocks);
    }

    [Fact]
    public void NewerSequence_NeverPreemptsAnOlderOne_ItWaitsInstead()
    {
        // Four blocks. "old" (15-token prompt) and "new" (16-token prompt) both fit
        // exactly. At the first decode step "new" needs a third block while "old"
        // still has room in its second: "new" must wait, not evict "old".
        var cfg = Config(numBlocks: 4);
        var pool = new BlockPool(cfg.NumBlocks, cfg.BlockSize, 0);
        var sched = new ContinuousBatchScheduler(cfg, pool, "fp-victim-order", NullLogger.Instance);
        var old = Sequence("old", 15, 8);
        var newer = Sequence("new", 16, 8);
        sched.Submit(old);
        sched.Submit(newer);

        var prefill = sched.Schedule();
        Assert.Equal(2, prefill.ScheduledWork.Count);
        foreach (var work in prefill.ScheduledWork)
        {
            work.Sequence.AdvanceComputedTokens(work.NumScheduledTokens);
            work.Sequence.AppendOutputToken(3);
        }
        Assert.Equal(0, pool.NumFreeBlocks);

        var decode = sched.Schedule();

        Assert.Empty(decode.PreemptedRequestIds);
        var scheduled = Assert.Single(decode.ScheduledWork);
        Assert.Same(old, scheduled.Sequence);
        Assert.Equal(SequenceStatus.Running, old.Status);
        Assert.Equal(SequenceStatus.Running, newer.Status);
        Assert.Equal(16, newer.NumComputedTokens);
    }

    [Fact]
    public void RandomWorkloads_NeverPreemptOlderWorkForNewer_AndAlwaysFinish()
    {
        // Staggered arrivals of mixed prompt/output lengths against small pools:
        // decode growth still needs preemption, but a victim is always younger
        // than every sequence the same step serves, and nothing livelocks.
        var rng = new Random(20260916);
        int totalPreemptions = 0;
        for (int trial = 0; trial < 300; trial++)
        {
            int numBlocks = rng.Next(6, 20);
            var cfg = new SchedulerConfig
            {
                MaxNumBatchedTokens = rng.Next(9, 40),
                MaxNumRunningSequences = rng.Next(2, 6),
                MaxPrefillChunkSize = rng.Next(2, 12),
                SoloPrefillChunkSize = 32,
                NumBlocks = numBlocks,
                BlockSize = BlockSize,
                EnablePrefixCaching = false,
                DecodeQuantumTokens = 1,
                StopRepetition = false,
            };
            var pool = new BlockPool(cfg.NumBlocks, cfg.BlockSize, 0);
            var sched = new ContinuousBatchScheduler(cfg, pool, $"fp-random-{trial}", NullLogger.Instance);
            int count = rng.Next(2, 7);
            var pending = new List<(int Step, SequenceState Seq)>();
            for (int i = 0; i < count; i++)
            {
                int capacity = numBlocks * BlockSize;
                int prompt = rng.Next(1, Math.Max(2, capacity / 2));
                int maxNew = rng.Next(1, Math.Max(2, capacity - prompt));
                pending.Add((rng.Next(0, 30), Sequence($"t{trial}-s{i}", prompt, maxNew)));
            }
            pending.Sort((a, b) => a.Step.CompareTo(b.Step));
            // Sn follows construction order; submit in that order too so rank == arrival.
            pending = pending.Select((p, i) => (p.Step, Sequence(p.Seq.RequestId, p.Seq.PromptTokens.Count, p.Seq.MaxNewTokens))).ToList();
            var all = pending.Select(p => p.Seq).ToList();

            int step = 0, next = 0;
            while (all.Any(s => !s.Status.IsFinished()))
            {
                Assert.True(step < 20000, $"trial {trial}: no completion after {step} steps (livelock)");
                while (next < pending.Count && pending[next].Step <= step)
                    sched.Submit(pending[next++].Seq);
                var blocksBefore = all.ToDictionary(s => s.RequestId, s => s.BlockTable.NumBlocks);
                var output = sched.Schedule();
                step++;
                totalPreemptions += output.PreemptedRequestIds.Count;
                // A preemption exists to give blocks to the sequence that needed them;
                // that sequence must outrank (be older than) the victim.
                var allocators = output.ScheduledWork
                    .Where(w => w.Sequence.BlockTable.NumBlocks > blocksBefore[w.Sequence.RequestId])
                    .Select(w => w.Sequence).ToList();
                foreach (string victimId in output.PreemptedRequestIds)
                {
                    var victim = all.Single(s => s.RequestId == victimId);
                    Assert.True(allocators.Any(a => a.Sn < victim.Sn),
                        $"trial {trial}: {victim.RequestId} (sn {victim.Sn}) was preempted for newer work " +
                        $"({string.Join(", ", allocators.Select(a => a.RequestId + " sn " + a.Sn))})");
                }
                foreach (var work in output.ScheduledWork)
                {
                    var seq = work.Sequence;
                    seq.AdvanceComputedTokens(work.NumScheduledTokens);
                    if (seq.NumComputedTokens < seq.NumTotalTokens) continue;
                    seq.AppendOutputToken(3);
                    if (seq.ShouldStopForLength())
                        sched.NotifyStop(seq, SequenceStatus.FinishedLengthCapped, "length", output);
                }
            }
            Assert.Equal(cfg.NumBlocks, pool.NumFreeBlocks);
        }
        // The workloads do exercise preemption (decode growth is never reserved).
        Assert.True(totalPreemptions > 0);
    }

    [Fact]
    public void OlderSequence_StillPreemptsTheNewestWhenItNeedsBlocks()
    {
        // Mirror image: the OLD sequence needs the block, so the newest yields.
        var cfg = Config(numBlocks: 4);
        var pool = new BlockPool(cfg.NumBlocks, cfg.BlockSize, 0);
        var sched = new ContinuousBatchScheduler(cfg, pool, "fp-victim-order-2", NullLogger.Instance);
        var old = Sequence("old", 16, 8);
        var newer = Sequence("new", 15, 8);
        sched.Submit(old);
        sched.Submit(newer);

        var prefill = sched.Schedule();
        foreach (var work in prefill.ScheduledWork)
        {
            work.Sequence.AdvanceComputedTokens(work.NumScheduledTokens);
            work.Sequence.AppendOutputToken(3);
        }

        var decode = sched.Schedule();

        Assert.Equal(new[] { "new" }, decode.PreemptedRequestIds);
        Assert.Same(old, Assert.Single(decode.ScheduledWork).Sequence);
        Assert.Equal(SequenceStatus.Preempted, newer.Status);
    }

    [Fact]
    public void NewerDecode_NeverEvictsAnOlderPrefill_ThatIsNotYetScheduled()
    {
        // Sn follows construction, not submission: build "old" first, submit it
        // second. "new" prefills alone and decodes; "old" is admitted while "new"
        // decodes and prefills in 8-token chunks (the mixed-step cap). Decode runs
        // first, so when "new" needs its third block "old" is not yet part of the
        // step - exactly where rank-blind victim selection used to evict the older
        // prefill for the newer decode.
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 64,
            MaxNumRunningSequences = 4,
            MaxPrefillChunkSize = 8,
            SoloPrefillChunkSize = 64,
            NumBlocks = 4,
            BlockSize = BlockSize,
            EnablePrefixCaching = false,
            DecodeQuantumTokens = 1,
            StopRepetition = false,
        };
        var pool = new BlockPool(cfg.NumBlocks, cfg.BlockSize, 0);
        var sched = new ContinuousBatchScheduler(cfg, pool, "fp-victim-prefill", NullLogger.Instance);
        var old = Sequence("old", 24, 8);
        var newer = Sequence("new", 7, 16);
        Assert.True(old.Sn < newer.Sn);

        void Apply(SchedulerOutput output)
        {
            foreach (var work in output.ScheduledWork)
            {
                var seq = work.Sequence;
                seq.AdvanceComputedTokens(work.NumScheduledTokens);
                if (seq.NumComputedTokens < seq.NumTotalTokens) continue;
                seq.AppendOutputToken(3);
                if (seq.ShouldStopForLength())
                    sched.NotifyStop(seq, SequenceStatus.FinishedLengthCapped, "length", output);
            }
        }

        sched.Submit(newer);
        Apply(sched.Schedule());
        sched.Submit(old);

        SchedulerOutput contested = null;
        for (int step = 0; step < 64 && contested == null; step++)
        {
            var output = sched.Schedule();
            if (output.PreemptedRequestIds.Count > 0)
                contested = output;
            Apply(output);
        }

        Assert.NotNull(contested);
        Assert.Equal(new[] { "new" }, contested.PreemptedRequestIds);
        Assert.Contains(contested.ScheduledWork, w => ReferenceEquals(w.Sequence, old));
    }

    [Fact]
    public void SharedPrefixBlocksInUse_DoNotCountAgainstAdmission()
    {
        // 12 blocks x 8. Each prompt is a 64-token shared prefix (8 blocks) plus 8
        // unique tokens. While "a" runs and holds the prefix, "b" adopts those 8
        // blocks and needs only its own tail: it must be admitted next to "a", not
        // held back as if its whole 9-block prompt had to come out of the free pool.
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 128,
            MaxNumRunningSequences = 4,
            MaxPrefillChunkSize = 128,
            SoloPrefillChunkSize = 128,
            NumBlocks = 12,
            BlockSize = BlockSize,
            EnablePrefixCaching = true,
            DecodeQuantumTokens = 1,
            StopRepetition = false,
        };
        var pool = new BlockPool(cfg.NumBlocks, cfg.BlockSize, 0);
        var sched = new ContinuousBatchScheduler(cfg, pool, "fp-shared-prefix", NullLogger.Instance);
        var prefix = Enumerable.Range(1, 64).ToList();
        SequenceState Shared(string id, int salt)
            => new(id, prefix.Concat(Enumerable.Range(1000 + salt, 8)).ToList(), 8, BlockSize, SamplingConfig.Greedy);
        var a = Shared("a", 0);
        var b = Shared("b", 100);

        sched.Submit(a);
        var first = sched.Schedule();
        foreach (var work in first.ScheduledWork)
        {
            int before = work.Sequence.NumComputedTokens;
            work.Sequence.AdvanceComputedTokens(work.NumScheduledTokens);
            sched.OnBlocksCommitted(work.Sequence, before);
            work.Sequence.AppendOutputToken(3);
        }
        Assert.Equal(72, a.NumComputedTokens);

        sched.Submit(b);
        var second = sched.Schedule();

        Assert.Empty(second.PreemptedRequestIds);
        Assert.Equal(SequenceStatus.Running, b.Status);
        Assert.Equal(64, b.PrefixCacheReusedTokens);
        Assert.Contains(second.ScheduledWork, w => ReferenceEquals(w.Sequence, b));
    }
    /// <summary>
    /// The shared-prefix discount is for pooled adoption only. Admission tries a
    /// retained fused holder first (Gemma 4 has both), and the executor backs that
    /// holder with ceil(lcp / BlockSize) NEW blocks for the whole prefix, not the
    /// blocks another running request already holds. A candidate that the discount
    /// admits but the holder serves takes blocks the running prompts still need.
    /// </summary>
    [Theory]
    [InlineData(false, true)]   // pooled adoption: the discount is real, "b" runs next to "a"
    [InlineData(true, false)]   // a retained holder serves "b": no discount, "b" waits
    public void SharedPrefixDiscount_DoesNotApply_WhenARetainedHolderServesThePrefix(bool fusedMatch, bool admitted)
    {
        // 20 blocks x 8. "a" is a 144-token prompt (64-token shared prefix + 80) that
        // prefills 8 tokens per contended step; "b" is the same prefix + 8 tokens.
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 16,
            MaxNumRunningSequences = 4,
            MaxPrefillChunkSize = 16,
            SoloPrefillChunkSize = 16,
            NumBlocks = 20,
            BlockSize = BlockSize,
            EnablePrefixCaching = true,
            DecodeQuantumTokens = 1,
            StopRepetition = false,
        };
        var pool = new BlockPool(cfg.NumBlocks, cfg.BlockSize, 0);
        var sched = new ContinuousBatchScheduler(cfg, pool, "fp-fused-discount", NullLogger.Instance);
        var prefix = Enumerable.Range(1, 64).ToList();
        var a = new SequenceState("a", prefix.Concat(Enumerable.Range(500, 80)).ToList(), 8, BlockSize, SamplingConfig.Greedy);
        var b = new SequenceState("b", prefix.Concat(Enumerable.Range(900, 8)).ToList(), 8, BlockSize, SamplingConfig.Greedy);

        int fusedAdoptions = 0;
        // Mirrors BatchExecutor.ComputeFusedContinuationLcp / TryAdoptFusedContinuation:
        // a holder of "b"'s conversation covers the 64-token prefix, and adopting it
        // allocates fresh placeholder blocks for the whole prefix.
        sched.AttachFusedCacheContinuation(
            seq => fusedMatch && seq.RequestId == "b" ? 64 : 0,
            (seq, lcp) =>
            {
                var blocks = pool.AllocateNew((lcp + BlockSize - 1) / BlockSize);
                if (blocks == null) return false;
                foreach (var block in blocks) seq.BlockTable.AppendBlock(block);
                seq.SetComputedTokensForPrefixAdoption(lcp);
                seq.PrefixCacheReusedTokens = lcp;
                fusedAdoptions++;
                return true;
            });

        void Apply(SchedulerOutput output)
        {
            foreach (var work in output.ScheduledWork)
            {
                int before = work.Sequence.NumComputedTokens;
                work.Sequence.AdvanceComputedTokens(work.NumScheduledTokens);
                sched.OnBlocksCommitted(work.Sequence, before);
            }
        }

        sched.Submit(a);
        while (a.NumComputedTokens < 64)
            Apply(sched.Schedule());
        Assert.Equal(64, a.NumComputedTokens);
        Assert.Equal(8, a.BlockTable.NumBlocks);

        // This step "a" takes its next 8 tokens (a 9th block): 11 blocks free, 9 more
        // still owed to "a"'s prompt, so 2 are available. "b" needs 9, or 1 when its 8
        // prefix blocks are shared with "a"; a holder instead takes 8 new ones.
        sched.Submit(b);
        var contested = sched.Schedule();

        Assert.Empty(contested.PreemptedRequestIds);
        if (admitted)
        {
            Assert.Equal(SequenceStatus.Running, b.Status);
            Assert.Equal(64, b.PrefixCacheReusedTokens);
            Assert.Equal(0, fusedAdoptions);
        }
        else
        {
            Assert.Equal(SequenceStatus.Waiting, b.Status);
            Assert.Equal(0, b.BlockTable.NumBlocks);
            Assert.Equal(0, fusedAdoptions);
            Assert.DoesNotContain(contested.ScheduledWork, w => ReferenceEquals(w.Sequence, b));
        }

        // Both finish and every block comes back. When the holder serves "b" it runs
        // after "a", so nothing is preempted; pooled sharing leaves no room for "a"'s
        // decode growth, which preempts "b" as designed (growth is never reserved).
        var all = new[] { a, b };
        for (int step = 0; step < 400 && all.Any(s => !s.Status.IsFinished()); step++)
        {
            var output = sched.Schedule();
            if (fusedMatch)
                Assert.Empty(output.PreemptedRequestIds);
            foreach (var work in output.ScheduledWork)
            {
                var seq = work.Sequence;
                int before = seq.NumComputedTokens;
                seq.AdvanceComputedTokens(work.NumScheduledTokens);
                sched.OnBlocksCommitted(seq, before);
                if (seq.NumComputedTokens < seq.NumTotalTokens) continue;
                seq.AppendOutputToken(3);
                if (seq.ShouldStopForLength())
                    sched.NotifyStop(seq, SequenceStatus.FinishedLengthCapped, "length", output);
            }
        }
        Assert.All(all, s => Assert.Equal(SequenceStatus.FinishedLengthCapped, s.Status));
        Assert.Equal(cfg.NumBlocks, pool.NumFreeBlocks);
    }

    /// <summary>The shared-prefix discount counts only the blocks the candidate would
    /// really adopt: a cache breakpoint, a different picture at the same position, a
    /// model that cannot continue past a media span, or another conversation's scope
    /// each stop adoption early, and every block past that point is new. Were the
    /// discount computed from the raw token match, "b" below would be admitted into a
    /// pool that cannot hold its prompt.</summary>
    [Theory]
    [InlineData("none", true)]
    [InlineData("breakpoint", false)]
    [InlineData("different-picture", false)]
    [InlineData("same-picture", true)]
    [InlineData("same-picture-no-media-reuse", false)]
    [InlineData("same-scope", true)]
    [InlineData("other-scope", false)]
    public void SharedPrefixDiscount_CountsOnlyBlocksAdoptionWouldTake(string variant, bool admitted)
    {
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 128,
            MaxNumRunningSequences = 4,
            // A one-block chunk fits the free pool whatever was adopted, so only the
            // capacity check (not the chunk's own allocation) can hold "b" back.
            MaxPrefillChunkSize = BlockSize,
            SoloPrefillChunkSize = 128,
            NumBlocks = 12,
            BlockSize = BlockSize,
            EnablePrefixCaching = true,
            DecodeQuantumTokens = 1,
            StopRepetition = false,
        };
        var pool = new BlockPool(cfg.NumBlocks, cfg.BlockSize, 0);
        var sched = new ContinuousBatchScheduler(cfg, pool, "fp-shared-prefix-limits", NullLogger.Instance,
            supportsReuseAcrossMediaSpan: variant != "same-picture-no-media-reuse");
        var prefix = Enumerable.Range(1, 64).ToList();
        bool media = variant.Contains("picture");
        bool scoped = variant.EndsWith("scope");
        SequenceState Shared(string id, int salt, string picture, IReadOnlyList<int> breakpoints, string scope)
            => new(id, prefix.Concat(Enumerable.Range(1000 + salt, 8)).ToList(), 8, BlockSize, SamplingConfig.Greedy,
                mediaSpans: picture == null ? null : new[] { new PromptMediaSpan(16, 32, picture) },
                cacheBreakpoints: breakpoints,
                sharedPrefixTokens: scoped ? 16 : 0,
                cacheScope: scope);
        var a = Shared("a", 0, media ? "picture-a" : null, null, scoped ? "chat-a" : null);
        var b = Shared("b", 100,
            variant == "different-picture" ? "picture-b" : media ? "picture-a" : null,
            variant == "breakpoint" ? new[] { 16 } : null,
            variant == "other-scope" ? "chat-b" : scoped ? "chat-a" : null);

        sched.Submit(a);
        // A shared-prefix boundary splits "a"'s prefill at token 16, so drive it until
        // its whole prompt is in the pool.
        for (int step = 0; step < 4 && a.NumComputedTokens < 72; step++)
        {
            foreach (var work in sched.Schedule().ScheduledWork)
            {
                int before = work.Sequence.NumComputedTokens;
                work.Sequence.AdvanceComputedTokens(work.NumScheduledTokens);
                sched.OnBlocksCommitted(work.Sequence, before);
                if (work.Sequence.NumComputedTokens == work.Sequence.NumTotalTokens)
                    work.Sequence.AppendOutputToken(3);
            }
        }
        Assert.Equal(72, a.NumComputedTokens);

        // 3 blocks are free; "b" needs 9. Adopting all 8 prefix blocks leaves 1 new
        // block; stopping at token 16 leaves 7, which must wait for "a".
        sched.Submit(b);
        var second = sched.Schedule();

        Assert.Empty(second.PreemptedRequestIds);
        if (admitted)
        {
            Assert.Equal(SequenceStatus.Running, b.Status);
            Assert.Equal(64, b.PrefixCacheReusedTokens);
        }
        else
        {
            Assert.Equal(SequenceStatus.Waiting, b.Status);
            Assert.DoesNotContain(second.ScheduledWork, w => ReferenceEquals(w.Sequence, b));
        }
    }
}
