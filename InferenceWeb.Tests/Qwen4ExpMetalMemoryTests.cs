// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>
/// Flash Next on Metal ran the GPU out of memory under Strata's benchmark (16K-token prompts that think, then four
/// concurrent requests) on a 48 GB M5 Pro: the expert tier left ~14 GB of the 38.3 GB working set and nothing bounded
/// how many concurrent requests' state (~1.1 GB each) or how large a prompt arena went beside it. Metal fails the
/// command buffer instead of the allocation, which no later graph recovers from. A new concurrent request now needs its
/// state free beside a headroom or waits (SequenceSlotUnavailableException), and a prompt arena that does not fit is
/// refused natively so the prompt runs in halves. These pin the admission rule; the live behaviour was checked on the
/// model (Strata's standard-v1 bundle after the engine benchmark on one server: no failure, 12/12 sanity).
/// </summary>
public class Qwen4ExpMetalMemoryTests
{
    private const long Gb = 1L << 30;
    private const long WorkingSet = 40L * Gb;

    [Fact]
    public void TheHeadroom_IsAGigabyte_OrAThirtySecondOfTheWorkingSet()
    {
        Assert.Equal(Gb, Qwen4ExpModel.MetalHeadroom(8 * Gb));
        Assert.Equal(WorkingSet / 32, Qwen4ExpModel.MetalHeadroom(WorkingSet));
    }

    [Fact]
    public void AnotherRequest_IsAdmitted_OnlyWhenItsStateLeavesTheHeadroom()
    {
        long need = 1100L << 20;   // a Flash Next request's state at a 32K context
        long headroom = Qwen4ExpModel.MetalHeadroom(WorkingSet);
        Assert.True(Qwen4ExpModel.FitsAnotherSequence(need + headroom, WorkingSet, need));
        Assert.False(Qwen4ExpModel.FitsAnotherSequence(need + headroom - 1, WorkingSet, need));
        // The free working set the failing run reached with three requests running.
        Assert.False(Qwen4ExpModel.FitsAnotherSequence(2156L << 20, WorkingSet, need));
        Assert.True(Qwen4ExpModel.FitsAnotherSequence(9 * Gb, WorkingSet, need));
    }
}
