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
}
