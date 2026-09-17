// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Linq;
using System.Reflection;

namespace InferenceWeb.Tests;

/// <summary>
/// The one-GGML-backend-per-process gate. A model test that constructs a GGML
/// backend other than the pinned one can only fail with "A different GGML backend
/// was already initialized"; the gate turns that into a visible skip in the lanes
/// that pin another backend, and never skips the lane that pins it.
/// </summary>
public sealed class GgmlBackendPinGateTests
{
    [Fact]
    public void PinnedBackendRuns_OtherGgmlBackendsSkip_NonGgmlBackendsNeverConflict()
    {
        BackendType pinned = TestGates.PinnedGgmlBackend;
        Assert.Null(TestGates.GgmlPinSkip(pinned));
        foreach (BackendType other in new[] { BackendType.GgmlCpu, BackendType.GgmlMetal, BackendType.GgmlCuda, BackendType.GgmlVulkan }
                     .Where(backend => backend != pinned))
        {
            string reason = TestGates.GgmlPinSkip(other);
            Assert.NotNull(reason);
            Assert.Contains("TS_TEST_GGML_BACKEND=", reason);
        }
        foreach (BackendType managed in new[] { BackendType.Cpu, BackendType.Cuda, BackendType.Mlx })
            Assert.Null(TestGates.GgmlPinSkip(managed));
    }

    [Fact]
    public void GgmlBackendNamedArgument_SetsTheSkipOnlyWhenNoEarlierGateDid()
    {
        BackendType other = TestGates.PinnedGgmlBackend == BackendType.GgmlCpu ? BackendType.GgmlCuda : BackendType.GgmlCpu;

        var unpinned = new ModelFactAttribute("TS_TEST_PIN_GATE_UNSET_" + nameof(GgmlBackendPinGateTests));
        string modelSkip = unpinned.Skip;
        Assert.NotNull(modelSkip);
        unpinned.GgmlBackend = other;
        Assert.Equal(modelSkip, unpinned.Skip);

        var fact = new ModelFactAttribute("TS_TEST_PIN_GATE_UNSET_" + nameof(GgmlBackendPinGateTests)) { Skip = null };
        fact.GgmlBackend = TestGates.PinnedGgmlBackend;
        Assert.Null(fact.Skip);
        fact.GgmlBackend = other;
        Assert.Equal(TestGates.GgmlPinSkip(other), fact.Skip);
    }

    [Fact]
    public void TestsThatHardCodeADifferentGgmlBackendDeclareIt()
    {
        // The model tests that construct a fixed GGML backend carry the gate, so a
        // lane pinned to another backend skips them instead of failing them.
        (string Type, string Method, BackendType Backend)[] declared =
        {
            (nameof(DeepSeekNativeRetentionFixtureTests), nameof(DeepSeekNativeRetentionFixtureTests.FailingPostCommitDiagnosticCannotUndoNativeOwnership), BackendType.GgmlCpu),
            (nameof(DeepSeekNativeRetentionFixtureTests), nameof(DeepSeekNativeRetentionFixtureTests.RetainedNativeHolderPreservesContinuationAndCanBeReclaimedWithoutASparePrimary), BackendType.GgmlCpu),
            (nameof(DeepSeek41DsparkIntegrationTests), nameof(DeepSeek41DsparkIntegrationTests.CpuAttachedBlockHead_EngagesAndPreservesSixteenGreedyTokens), BackendType.GgmlCpu),
            (nameof(DeepSeek41DsparkIntegrationTests), nameof(DeepSeek41DsparkIntegrationTests.VerifyRewindAndTwoSlots_PreserveAcceptedPrefixAndUnrelatedContinuation), BackendType.GgmlCpu),
            (nameof(Gemma4CacheResidencyTests), nameof(Gemma4CacheResidencyTests.InitializeResidentCache_DoesNotRestorePreviousDeviceContents), BackendType.GgmlCuda),
        };
        foreach (var (typeName, methodName, backend) in declared)
        {
            MethodInfo method = typeof(GgmlBackendPinGateTests).Assembly.GetTypes()
                .Single(type => type.Name == typeName).GetMethod(methodName)!;
            object gate = method.GetCustomAttributes().Single(attribute => attribute is FactAttribute);
            PropertyInfo property = gate.GetType().GetProperty("GgmlBackend");
            Assert.True(property != null, $"{typeName}.{methodName}: its gate has no GgmlBackend argument");
            Assert.Equal(backend, (BackendType)property.GetValue(gate)!);
        }
    }
}
