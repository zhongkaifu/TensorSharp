// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

/// <summary>
/// D01: admission and execution read one route decision. <see cref="ExecutionPlanner.PlanAdmission"/> says where a request's
/// state lives and which reused states it can resume; every step plan the executor then makes for it reads that state on
/// the same route (a slot or holder by the per-request paths, the primary by the per-sequence path, pages by the batched
/// route), so no reuse admission promises is one execution cannot read.
/// </summary>
public sealed class AdmissionPlanTests
{
    private static IEnumerable<ExecutionCapabilities> Capabilities()
    {
        foreach (bool unified in new[] { false, true })
        foreach (bool batched in new[] { false, true })
        foreach (bool forward in new[] { false, true })
        foreach (bool perSequence in new[] { false, true })
        foreach (bool media in new[] { false, true })
        {
            if (!batched && (forward || perSequence || media)) continue;   // the batched surface needs the contract
            yield return new ExecutionCapabilities
            {
                SupportsUnifiedBatching = unified,
                SupportsBatchedPagedAttention = batched,
                BatchedForwardAvailable = forward,
                SupportsPerSequenceFusedForward = perSequence,
                SupportsBatchedMultimodal = media,
                SupportsKvStateSnapshot = true,
                SupportsCrossSequenceKvReuse = true,
            };
        }
    }

    private static IEnumerable<ExecutionOptions> Options()
    {
        foreach (bool disabled in new[] { false, true })
        foreach (bool perSeq in new[] { false, true })
            yield return ExecutionOptions.Default with { BatchedPathDisabled = disabled, PerSeqFusedEnabled = perSeq };
    }

    [Fact]
    public void EveryStepReadsTheStateWhereAdmissionPutIt()
    {
        int combinations = 0;
        foreach (ExecutionCapabilities caps in Capabilities())
        foreach (ExecutionOptions options in Options())
        {
            AdmissionPlan admission = ExecutionPlanner.PlanAdmission(caps, options);
            // A concurrent step (two requests, the one where the paths differ).
            ExecutionPlan step = ExecutionPlanner.PlanStep(caps, options, new SchedulerConfig(), new ExecutionStepFeatures { SequenceCount = 2 });
            ExecutionPathKind first = step.Candidates[0];
            if (admission.Route == AdmissionRoute.Slots)
                Assert.True(first is ExecutionPathKind.Unified or ExecutionPathKind.PerSequenceFused,
                    $"{caps} {options}: a slot route planned {first}");
            else
                Assert.DoesNotContain(ExecutionPathKind.PerSequenceFused, step.Candidates);
            if (first == ExecutionPathKind.Unified)
                Assert.Equal(AdmissionRoute.Slots, admission.Route);
            // A paged end state is resumable only where the batched route runs.
            if (admission.PagedEndStatesReadable)
                Assert.True(caps.BatchedForwardAvailable && !options.BatchedPathDisabled);
            combinations++;
        }
        Assert.True(combinations > 20);
    }

    [Fact]
    public void AUnifiedModelReadsItsSlotsWhateverTheSwitchesSay()
    {
        var caps = new ExecutionCapabilities { SupportsUnifiedBatching = true };
        foreach (ExecutionOptions options in Options())
        {
            AdmissionPlan admission = ExecutionPlanner.PlanAdmission(caps, options);
            Assert.Equal(AdmissionRoute.Slots, admission.Route);
            Assert.True(admission.EndStatesReadable(pagedEndStates: false));
            Assert.False(admission.EndStatesReadable(pagedEndStates: true));
            Assert.False(admission.ReplansOnBatchedPaged(hasMedia: false));
        }
    }

    [Fact]
    public void APrimaryRouteReplansOnTheBatchedRouteOnlyWhereItRuns()
    {
        var caps = new ExecutionCapabilities { SupportsBatchedPagedAttention = true, BatchedForwardAvailable = true };
        AdmissionPlan plan = ExecutionPlanner.PlanAdmission(caps, ExecutionOptions.Default);
        Assert.Equal(AdmissionRoute.Primary, plan.Route);
        Assert.True(plan.ReplansOnBatchedPaged(hasMedia: false));
        Assert.False(plan.ReplansOnBatchedPaged(hasMedia: true));   // no batched multimodal
        Assert.True(ExecutionPlanner.PlanAdmission(caps with { SupportsBatchedMultimodal = true }, ExecutionOptions.Default)
            .ReplansOnBatchedPaged(hasMedia: true));
        Assert.False(ExecutionPlanner.PlanAdmission(caps, ExecutionOptions.Default with { BatchedPathDisabled = true })
            .ReplansOnBatchedPaged(hasMedia: false));
    }

    [Fact]
    public void ThePerSequenceSwitchesDecideALegacyModelsRoute()
    {
        var caps = new ExecutionCapabilities { SupportsBatchedPagedAttention = true, SupportsPerSequenceFusedForward = true };
        Assert.Equal(AdmissionRoute.Slots, ExecutionPlanner.PlanAdmission(caps, ExecutionOptions.Default).Route);
        Assert.Equal(AdmissionRoute.Primary,
            ExecutionPlanner.PlanAdmission(caps, ExecutionOptions.Default with { PerSeqFusedEnabled = false }).Route);
        Assert.Equal(AdmissionRoute.Primary,
            ExecutionPlanner.PlanAdmission(caps, ExecutionOptions.Default with { BatchedPathDisabled = true }).Route);
    }
}
