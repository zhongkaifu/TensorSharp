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
        // The tests that construct a fixed GGML backend carry the gate, so a lane
        // pinned to another backend skips them instead of failing them.
        (string Type, string Method, BackendType Backend)[] declared =
        {
            (nameof(DeepSeekNativeRetentionFixtureTests), nameof(DeepSeekNativeRetentionFixtureTests.FailingPostCommitDiagnosticCannotUndoNativeOwnership), BackendType.GgmlCpu),
            (nameof(DeepSeekNativeRetentionFixtureTests), nameof(DeepSeekNativeRetentionFixtureTests.RetainedNativeHolderPreservesContinuationAndCanBeReclaimedWithoutASparePrimary), BackendType.GgmlCpu),
            (nameof(DeepSeek41DsparkIntegrationTests), nameof(DeepSeek41DsparkIntegrationTests.CpuAttachedBlockHead_EngagesAndPreservesSixteenGreedyTokens), BackendType.GgmlCpu),
            (nameof(DeepSeek41DsparkIntegrationTests), nameof(DeepSeek41DsparkIntegrationTests.VerifyRewindAndTwoSlots_PreserveAcceptedPrefixAndUnrelatedContinuation), BackendType.GgmlCpu),
            (nameof(Gemma4CacheResidencyTests), nameof(Gemma4CacheResidencyTests.InitializeResidentCache_DoesNotRestorePreviousDeviceContents), BackendType.GgmlCuda),
            // Weight-free tests that build ggml-cpu themselves ([GgmlFact]/[GgmlTheory]).
            (nameof(GgmlCopyDtypeTests), nameof(GgmlCopyDtypeTests.Copy_Float16_ContiguousTensors_RoundTripsExactBits), BackendType.GgmlCpu),
            (nameof(GgmlCopyDtypeTests), nameof(GgmlCopyDtypeTests.Copy_Float16_NarrowedKvCacheLayout_PreservesEachHeadIndependently), BackendType.GgmlCpu),
            (nameof(GgmlCopyDtypeTests), nameof(GgmlCopyDtypeTests.Copy_CrossDtype_Throws), BackendType.GgmlCpu),
            (nameof(GgmlCopyStridedFloat32Tests), nameof(GgmlCopyStridedFloat32Tests.Copy_NarrowOnInnerDim_MatchesElementwiseReference), BackendType.GgmlCpu),
            (nameof(GgmlCopyStridedFloat32Tests), nameof(GgmlCopyStridedFloat32Tests.Copy_NarrowOnMiddleDim_KvCacheResizeLayout_MatchesReference), BackendType.GgmlCpu),
            (nameof(GgmlCopyStridedFloat32Tests), nameof(GgmlCopyStridedFloat32Tests.Copy_TransposedView_StaysOnTheElementPath_AndIsCorrect), BackendType.GgmlCpu),
            (nameof(GgmlCopyStridedFloat32Tests), nameof(GgmlCopyStridedFloat32Tests.Copy_NarrowedOuterDim_ContiguousInner_MatchesReference), BackendType.GgmlCpu),
            (nameof(GgmlCopyStridedFloat32Tests), nameof(GgmlCopyStridedFloat32Tests.Copy_ContiguousToContiguous_IsUnchanged), BackendType.GgmlCpu),
            (nameof(Glm5NextNativeTensorParallelTests), nameof(Glm5NextNativeTensorParallelTests.NativeLoader_AcceptsGlm5NextWithTwoAlignedTpRanks), BackendType.GgmlCpu),
            (nameof(Glm5NextNativeTensorParallelTests), nameof(Glm5NextNativeTensorParallelTests.NativeLoader_RejectsTpPartitionThatCutsAQuantizationHeadGroup), BackendType.GgmlCpu),
            (nameof(Glm5NextSpeculativeRollbackTests), nameof(Glm5NextSpeculativeRollbackTests.NGramSpeculativeGreedy_MatchesPlainGreedy_AndRollsBack_GgmlCpu), BackendType.GgmlCpu),
            (nameof(Glm5NextSpeculativeRollbackTests), nameof(Glm5NextSpeculativeRollbackTests.EveryWindowPartiallyRejected_StillMatchesPlainGreedy_GgmlCpu), BackendType.GgmlCpu),
            (nameof(Glm5NextSpeculativeRollbackTests), nameof(Glm5NextSpeculativeRollbackTests.SnapshotVerifyRestoreRewind_EqualsAPlainDecodeOfTheAcceptedPrefix_GgmlCpu), BackendType.GgmlCpu),
            (nameof(Glm5NextSpeculativeRollbackTests), nameof(Glm5NextSpeculativeRollbackTests.BoundSlots_AbaRollbackPreservesEachContinuation), BackendType.GgmlCpu),
            (nameof(Glm5NextSpeculativeRollbackTests), nameof(Glm5NextSpeculativeRollbackTests.ManagedAndNativeSpecForward_AgreeRowForRow), BackendType.GgmlCpu),
            (nameof(Glm5NextSpeculationEligibilityTests), nameof(Glm5NextSpeculationEligibilityTests.ActualKdaCheckpoint_IsEligibleForNgram_OnTheRecurrentContract_GgmlCpu), BackendType.GgmlCpu),
            (nameof(Glm5NextSpeculationEligibilityTests), nameof(Glm5NextSpeculationEligibilityTests.ArbitraryRewind_IsRefusedWithoutMutatingTheContinuation_GgmlCpu), BackendType.GgmlCpu),
            (nameof(Glm5NextSpeculationEligibilityTests), nameof(Glm5NextSpeculationEligibilityTests.SchedulerRequestedNgram_ArmsAndPreservesThePlainStream_GgmlCpu), BackendType.GgmlCpu),
            (nameof(GlmTruncateRefusalTests), nameof(GlmTruncateRefusalTests.GlmDsa_NativeRefusal_IsReportedAndLeavesTheHead), BackendType.GgmlCpu),
            (nameof(GlmTruncateRefusalTests), nameof(GlmTruncateRefusalTests.GlmDsa_AcceptedRewind_MovesTheHead_AndContinuesExactly), BackendType.GgmlCpu),
            (nameof(GlmTruncateRefusalTests), nameof(GlmTruncateRefusalTests.Glm5Next_NativeRefusesAMidSequenceRewind_AndTheTrunkIsUntouched), BackendType.GgmlCpu),
            (nameof(GlmTruncateRefusalTests), nameof(GlmTruncateRefusalTests.Glm5Next_RewindToTheHeadOrToZero_IsStillAccepted), BackendType.GgmlCpu),
            (nameof(GlmTruncateRefusalTests), nameof(GlmTruncateRefusalTests.Glm_VoidTruncate_ThrowsOnARefusal_InsteadOfReturningWithAStaleHead), BackendType.GgmlCpu),
            (nameof(NemotronConvTests), nameof(NemotronConvTests.NativeMamba2Prefill_MatchesManagedReferenceOnCpu), BackendType.GgmlCpu),
            (nameof(NemotronConvTests), nameof(NemotronConvTests.NativeMamba2Decode_PersistentStateMatchesManagedReferenceOnCpu), BackendType.GgmlCpu),
            (nameof(NemotronAudioEncoderTests), nameof(NemotronAudioEncoderTests.OfficialParakeetAndProjection_AllRowsMatch_GgmlCpu), BackendType.GgmlCpu),
            (nameof(KvStateFingerprintNonEmptyTests), nameof(KvStateFingerprintNonEmptyTests.Glm_SyntheticCheckpoints_ReportDistinctNonEmptyFingerprints), BackendType.GgmlCpu),
        };
        foreach (var (typeName, methodName, backend) in declared)
        {
            MethodInfo method = typeof(GgmlBackendPinGateTests).Assembly.GetTypes()
                .Single(type => type.Name == typeName).GetMethod(methodName)!;
            object gate = method.GetCustomAttributes().Single(attribute => attribute is FactAttribute);
            PropertyInfo property = gate.GetType().GetProperty("GgmlBackend");
            Assert.True(property != null, $"{typeName}.{methodName}: its gate has no GgmlBackend argument");
            Assert.Equal(backend, (BackendType)property.GetValue(gate)!);
            // And the gate acts on it: every lane that pins another backend skips.
            if (TestGates.GgmlPinSkip(backend) != null)
                Assert.True(((FactAttribute)gate).Skip != null, $"{typeName}.{methodName}: runs in a lane pinned to {TestGates.PinnedGgmlBackend}");
        }
    }

    [Fact]
    public void GgmlFactAndTheory_SkipExactlyWhenAnotherBackendIsPinned_AndCarryTheRequiresTrait()
    {
        foreach (BackendType backend in new[] { BackendType.GgmlCpu, BackendType.GgmlMetal, BackendType.GgmlCuda, BackendType.GgmlVulkan })
        {
            var fact = new GgmlFactAttribute(backend);
            Assert.Equal(backend, fact.GgmlBackend);
            Assert.Equal(TestGates.GgmlPinSkip(backend), fact.Skip);
            Assert.Equal(backend.ToString(), fact.RequiresValue);

            var theory = new GgmlTheoryAttribute(backend);
            Assert.Equal(backend, theory.GgmlBackend);
            Assert.Equal(TestGates.GgmlPinSkip(backend), theory.Skip);
            Assert.Equal(backend.ToString(), theory.RequiresValue);
        }
        Assert.Null(new GgmlFactAttribute(TestGates.PinnedGgmlBackend).Skip);
    }
}
