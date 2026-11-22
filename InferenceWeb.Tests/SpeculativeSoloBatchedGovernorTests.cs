// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The cost governor judges a lone request's steps on the unified runtime's batched path, as it judges the solo
// DecodeStep: a step serving one sequence is that sequence's solo step. Ungoverned, it drafted every step whatever the
// drafts cost (Muse-Glimmer-30B's DFlash on an M5 Pro, the runtime's unified step: an essay at 20.9 tok/s against 21.9 plain,
// where the legacy model's governed solo step parked). On a scripted clock; no test here measures real time.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;
using Xunit;

namespace InferenceWeb.Tests;

public partial class SpeculativeExecutionTests
{
    /// <summary>
    /// One request on the batched step: a drafter whose drafts cost four verifies and are always rejected is parked by the
    /// governor (its parked steps verify the anchor alone, so the drafter's state stays in step), and one whose drafts are
    /// cheap and kept is never parked. Either way the stream is the trunk's own.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void SoloBatchedStep_TheGovernorParksALosingDrafterAndKeepsAWinningOne(bool losing)
    {
        const long verify = 100_000, commit = 1_000;
        const int steps = 120, prompt = 6;
        var clock = new SimulatedClock();
        var model = new FakeBatchedSpeculativeModel
        {
            ResumesAfterGap = true, Clock = clock, VerifyCost = verify, CommitCost = commit,
            DraftCost = losing ? verify : verify / 100,
        };
        if (losing)
            for (int p = 0; p < prompt + steps * 5 + 8; p++) model.DraftWrongPositions.Add(p);
        var options = new SpeculationOptions
        {
            Enabled = true, SpeculatorName = SpeculatorRegistry.DraftHead, MaxDraftTokens = 4, MaxDraftTokensExplicit = true,
            MinDraftProb = 0f,
        };
        var pool = new BlockPool(256, BlockSize, BlockSize * 4);
        // The fake's stream repeats every 64 tokens: no repetition stop.
        var scheduler = new ContinuousBatchScheduler(new SchedulerConfig { Speculation = options, StopRepetition = false }, pool);
        var executor = new BatchExecutor(model, pool, scheduler) { Timestamp = clock.Read };
        var seq = new SequenceState("solo", Enumerable.Range(1, prompt).ToList(), steps * 6, BlockSize, SamplingConfig.Greedy);
        foreach (var block in pool.AllocateNew(200)!) seq.BlockTable.AppendBlock(block);
        seq.AdvanceComputedTokens(prompt);
        seq.Status = SequenceStatus.Running;
        seq.PendingDeviceToken = model.ExpectedNext(prompt - 1);
        seq.PendingDevicePosition = prompt;
        model.Start("solo", prompt);
        var step = typeof(BatchExecutor).GetMethod("TryBatchedSpeculativeDecode", BindingFlags.Instance | BindingFlags.NonPublic)!;

        var draftedAt = new List<bool>();
        for (int i = 0; i < steps; i++)
        {
            int before = model.DraftStepCalls;
            var results = new List<SequenceStepResult>();
            Assert.True((bool)step.Invoke(executor, new object[] { new List<ScheduledSequenceWork> { new(seq, 1, false, false) }, results })!);
            Assert.All(results, r => Assert.Null(r.Error));
            draftedAt.Add(model.DraftStepCalls > before);
        }

        // The trunk's stream whatever the drafter did.
        for (int t = 0; t < seq.OutputTokens.Count; t++)
            Assert.Equal(model.ExpectedNext(prompt - 1 + t), seq.OutputTokens[t]);
        Assert.Empty(model.ProtocolViolations);
        SpeculationStats stats = seq.SpecStats;
        int draftedSteps = draftedAt.Count(d => d);
        if (losing)
        {
            Assert.True(stats.GovernorLosses > 0, "the governor never judged the losing drafter");
            Assert.True(stats.ParkedSteps > steps / 2, $"parked {stats.ParkedSteps} of {steps} steps");
            Assert.True(draftedSteps < steps / 4, $"the losing drafter ran on {draftedSteps} of {steps} steps");
            // A parked step verified its anchor alone: one row, one token.
            Assert.Equal(steps, seq.OutputTokens.Count);
        }
        else
        {
            Assert.Equal(0L, stats.ParkedSteps);
            Assert.True(stats.GovernorWins > 0, "the governor never judged the winning drafter");
            Assert.True(stats.TokensAccepted > steps * 3, $"accepted {stats.TokensAccepted} drafts over {steps} steps");
        }
    }
}
