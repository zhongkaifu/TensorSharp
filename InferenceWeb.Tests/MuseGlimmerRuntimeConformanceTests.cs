// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Threading.Tasks;
using TensorSharp.Runtime.Speculative;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Muse-Glimmer's dedicated conformance suite (<see cref="SharedRuntimeConformanceTests.DedicatedSuites"/>): one test per
/// mechanism its <c>ServingContract.SharedRuntime</c> claims, each run by the family's own synthetic suites on fixtures whose
/// sliding-window rings lap (the shared script's fixtures have no rings): placements on simulated devices
/// (<see cref="MuseGlimmerRuntimeTests"/>), batched slots, continuous batching and checkpoints
/// (<see cref="MuseGlimmerRuntimeServingTests"/>), speculation through the engine (<see cref="MuseGlimmerRuntimeSpeculationTests"/>)
/// and whole-forward graphs against the per-op path (both held to the frozen legacy logits). Muse-Glimmer has no routed experts.
/// </summary>
public sealed class MuseGlimmerRuntimeConformanceTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData(2, 1)]
    [InlineData(1, 2)]
    [InlineData(2, 2)]
    public void PlacementMatchesOneDevice(int tp, int stages)
    {
        using var suite = new MuseGlimmerRuntimeTests(output);
        suite.PlacementsOnSimulatedDevicesGiveTheOneDeviceLogits(tp, stages, graphs: true);
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 1)]
    [InlineData(1, 2)]
    public void SlotBatchesMatchEachSequenceAlone(int tp, int stages)
    {
        using var suite = new MuseGlimmerRuntimeServingTests(output);
        suite.SlotBatchesOverRingsMatchEachSequenceAlone("managed", tp, stages);
    }

    [Fact]
    public async Task ConcurrentRequestsAreBatched()
    {
        using var suite = new MuseGlimmerRuntimeServingTests(output);
        await suite.ConcurrentRequestsOverRingsMatchSequentialOnes("managed");
    }

    [Fact]
    public void CheckpointsContinueExactly()
    {
        using var suite = new MuseGlimmerRuntimeServingTests(output);
        suite.CheckpointsOfWrappedRingsContinueExactly("managed");
    }

    [Theory]
    [InlineData(SpeculatorRegistry.Block)]
    [InlineData(SpeculatorRegistry.NGram)]
    public async Task SpeculativeDecodingMatchesPlainGreedy(string speculator)
    {
        using var suite = new MuseGlimmerRuntimeSpeculationTests(output);
        await suite.EngineSpeculationMatchesPlain("managed", speculator);
    }

    [Theory]
    [InlineData("untied/decode")]
    [InlineData("untied/ring-wrap")]
    public void WholeForwardGraphMatchesThePerOpPath(string caseName)
    {
        // Both paths are held to the same frozen legacy logits within the parity bound, so they are within twice it of each other.
        using var suite = new MuseGlimmerRuntimeTests(output);
        suite.RuntimeOnTheManagedBackendGivesTheFrozenLegacyLogits(caseName, "graphs");
        suite.RuntimeOnTheManagedBackendGivesTheFrozenLegacyLogits(caseName, "per-op");
    }
}
