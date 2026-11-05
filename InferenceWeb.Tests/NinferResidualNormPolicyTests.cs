// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using Pass = TensorSharp.Cuda.Q35CudaEngine.NinferLayerPass;
using State = TensorSharp.Cuda.Q35CudaEngine.NinferResidualState;
using Route = TensorSharp.Cuda.Q35CudaEngine.NinferNormRoute;

namespace InferenceWeb.Tests;

public sealed class NinferResidualNormPolicyTests
{
    private static bool Plan(string? option, bool qualified = true, int sms = 170, int context = 40960,
        int rows = 2048, bool verification = true, bool graphs = true, bool gemm8 = true,
        bool splitK = true, bool fi = false, bool exact = false, bool allDowns = true)
        => Q35CudaEngine.PlanNinferResidualNorm(true, qualified, sms, context, rows, verification,
            graphs, gemm8, splitK, fi, exact, option!, allDowns);

    [Theory]
    [InlineData(null, true)] [InlineData("0", false)] [InlineData("1", true)]
    public void QualifiedProfileUsesAutomaticResidualNormWithExplicitOptOut(string? option, bool expected)
        => Assert.Equal(expected, Plan(option));

    [Theory]
    [InlineData(40960)] [InlineData(65536)] [InlineData(131072)] [InlineData(262144)]
    public void OnlyTheTwoConfiguredContextProfilesCanUseTheSameT8Math(int context)
    {
        Assert.True(Plan(null, context: context)); Assert.True(Plan("1", context: context));
        Assert.False(Plan("0", context: context));
    }

    [Theory]
    [InlineData(32768)] [InlineData(65535)] [InlineData(65537)]
    public void OtherConfiguredContextsRemainUnsupported(int context)
    {
        Assert.False(Plan(null, context: context)); Assert.False(Plan("0", context: context));
        Assert.Throws<NotSupportedException>(() => Plan("1", context: context));
    }

    [Theory]
    [InlineData("")] [InlineData("true")] [InlineData(" 1")] [InlineData("01")] [InlineData("2")]
    public void InvalidNinferOptionCannotSilentlyFallBack(string option)
        => Assert.Throws<ArgumentException>(() => Plan(option, qualified: false));

    [Fact]
    public void UnmeasuredProfilesAndProjectionOverridesAreRefused()
    {
        Action[] requests = {
            () => Plan("1", qualified: false), () => Plan("1", sms: 169), () => Plan("1", sms: 171),
            () => Plan("1", context: 40959), () => Plan("1", context: 65535), () => Plan("1", rows: 7),
            () => Plan("1", verification: false), () => Plan("1", graphs: false), () => Plan("1", gemm8: false),
            () => Plan("1", splitK: false), () => Plan("1", fi: true), () => Plan("1", exact: true),
            () => Plan("1", allDowns: false)
        };
        foreach (var request in requests) Assert.Throws<NotSupportedException>(request);
        Assert.False(Plan(null, qualified: false, graphs: false, fi: true));
        Assert.False(Plan("0", qualified: false, graphs: false, fi: true));
    }

    [Fact]
    public void AutomaticSelectionDeclinesEveryUnmeasuredProfileWithoutChangingOtherFlags()
    {
        bool[] choices = { Plan(null, qualified: false), Plan(null, sms: 169), Plan(null, sms: 171),
            Plan(null, context: 40959), Plan(null, context: 65535), Plan(null, rows: 7),
            Plan(null, verification: false), Plan(null, graphs: false), Plan(null, gemm8: false),
            Plan(null, splitK: false), Plan(null, fi: true), Plan(null, exact: true),
            Plan(null, allDowns: false) };
        Assert.All(choices, choice => Assert.False(choice));
    }

    [Theory]
    [InlineData("ts_q35_gemm_w4a4_8_residual_bf16", 1)]
    [InlineData("ts_q35_ninfer_plain_norm_f32", 2)]
    public void AnOlderMainImageFallsBackOnlyBeforeEitherEntryIsPublished(string missing, int expectedReads)
    {
        int reads = 0;
        IntPtr Resolve(string symbol)
        {
            reads++;
            if (symbol == missing) throw new CudaException(500, "missing entry");
            return (IntPtr)123;
        }
        Assert.False(Q35Kernels.TryResolveNinferResidualNormEntries(false, Resolve, out var fused, out var plain));
        Assert.Equal(expectedReads, reads);
        Assert.Equal(IntPtr.Zero, fused); Assert.Equal(IntPtr.Zero, plain);
        var error = Assert.Throws<NotSupportedException>(() => Q35Kernels.TryResolveNinferResidualNormEntries(true, Resolve, out _, out _));
        Assert.IsType<CudaException>(error.InnerException);
    }

    [Theory]
    [InlineData(2)] [InlineData(201)] [InlineData(700)] [InlineData(719)]
    public void AutomaticSelectionNeverSwallowsAllocationContextOrExecutionErrors(int errorCode)
        => Assert.Throws<CudaException>(() => Q35Kernels.TryResolveNinferResidualNormEntries(false,
            _ => throw new CudaException(errorCode, "fatal"), out _, out _));

    [Fact]
    public void BothOriginalEntryAddressesResolveInOrderWithoutCudaOrPolicyMutation()
    {
        var names = new List<string>();
        Assert.True(Q35Kernels.TryResolveNinferResidualNormEntries(false, name =>
        { names.Add(name); return (IntPtr)names.Count; }, out var fused, out var plain));
        Assert.Equal(new[] { "ts_q35_gemm_w4a4_8_residual_bf16", "ts_q35_ninfer_plain_norm_f32" }, names);
        Assert.Equal((IntPtr)1, fused); Assert.Equal((IntPtr)2, plain);
        Assert.Throws<InvalidOperationException>(() => Q35Kernels.TryResolveNinferResidualNormEntries(false,
            _ => throw new InvalidOperationException("malformed ABI"), out _, out _));
    }

    [Theory]
    [InlineData(null)] [InlineData("0")] [InlineData("1")] [InlineData("invalid")]
    public void GgufIgnoresTheNewOptionEntirely(string? option)
        => Assert.False(Q35CudaEngine.PlanNinferResidualNorm(false, false, 0, 0, 0,
            false, false, false, false, false, false, option!));

    private static Pass Select(bool enabled = true, int groups = 1, int window = 8, int lanes = 1,
        bool verify = true, bool counts = false, bool embeds = false, bool taps = false, bool gguf = false, int first = 0)
        => Q35CudaEngine.SelectNinferResidualPass(enabled, groups, window, lanes, verify, counts, embeds, taps, gguf, first);

    [Fact]
    public void OnlyOneCompleteT8VerifyRequestsTheNewPass()
    {
        Assert.Equal(Pass.FusedResidualNorm, Select());
        foreach (var pass in new[] { Select(enabled: false), Select(groups: 2), Select(window: 1), Select(window: 7),
            Select(window: 9), Select(lanes: 2), Select(verify: false), Select(counts: true), Select(embeds: true),
            Select(taps: true), Select(gguf: true), Select(first: 1) }) Assert.Equal(Pass.Legacy, pass);
        var slots = new Q35CudaEngine.Slot[1]; var tokens = new int[8];
        Assert.Equal(Pass.Legacy, new Q35CudaEngine.RowGroup(slots, 8, true, null!, tokens).PassMode);
        Assert.Equal(Pass.FusedResidualNorm, new Q35CudaEngine.RowGroup(slots, 8, true, null!, tokens, passMode: Pass.FusedResidualNorm).PassMode);
        Assert.Throws<ArgumentException>(() => new Q35CudaEngine.RowGroup(slots, 8, false, null!, tokens, passMode: Pass.FusedResidualNorm));
        Assert.Throws<ArgumentException>(() => new Q35CudaEngine.RowGroup(slots, 8, true, new[] { 8 }, tokens, passMode: Pass.FusedResidualNorm));
        Assert.Throws<ArgumentException>(() => new Q35CudaEngine.RowGroup(slots, 8, true, null!, tokens, decode: true, passMode: Pass.FusedResidualNorm));
    }

    [Theory]
    [InlineData(5120, 6144)] [InlineData(5120, 17408)]
    public void ActualA4ShapeIsRequiredForAFusedResidual(int n, int k)
    {
        Assert.True(Q35CudaEngine.NinferResidualMatrixEligible(CudaNinferProjectionFormat.NvFp4Planes, true, 8, n, k));
        Assert.False(Q35CudaEngine.NinferResidualMatrixEligible(CudaNinferProjectionFormat.BFloat16, true, 8, n, k));
        Assert.False(Q35CudaEngine.NinferResidualMatrixEligible(CudaNinferProjectionFormat.NvFp4Planes, false, 8, n, k));
        Assert.False(Q35CudaEngine.NinferResidualMatrixEligible(CudaNinferProjectionFormat.NvFp4Planes, true, 7, n, k));
        Assert.False(Q35CudaEngine.NinferResidualMatrixEligible(CudaNinferProjectionFormat.NvFp4Planes, true, 9, n, k));
        Assert.False(Q35Kernels.NinferResidualProjectionShape(8, n, k + 256));
    }

    [Fact]
    public void InitialLegacyAndActuallyFusedStatesHaveDistinctNormSemantics()
    {
        Assert.Equal(Route.LegacyInitial, Q35CudaEngine.PlanNinferResidualNormalization(State.InitialNoDelta, true, false));
        Assert.Equal(Route.LegacyDelta, Q35CudaEngine.PlanNinferResidualNormalization(Q35CudaEngine.NinferProjectedResidualState(false), true, true));
        Assert.Equal(Route.ReadOnlyCanonical, Q35CudaEngine.PlanNinferResidualNormalization(Q35CudaEngine.NinferProjectedResidualState(true), true, true));
        Assert.Throws<InvalidOperationException>(() => Q35CudaEngine.PlanNinferResidualNormalization(State.InitialNoDelta, true, true));
        Assert.Throws<InvalidOperationException>(() => Q35CudaEngine.PlanNinferResidualNormalization(State.FusedCanonical, false, false));
        Assert.Throws<ArgumentOutOfRangeException>(() => Q35CudaEngine.PlanNinferResidualNormalization((State)99, true, false));
        // Old null-delta AddNorm adds +0, which can canonicalize signed zero.
        // Selecting the readonly route for the initial embedding would differ.
        float negativeZero = BitConverter.UInt32BitsToSingle(0x80000000u);
        Assert.NotEqual(BitConverter.SingleToInt32Bits(negativeZero), BitConverter.SingleToInt32Bits(negativeZero + 0.0f));
    }

    [Fact]
    public void EveryCompletedFusedDownMakesTheNextInputAndFinalNormReadonly()
    {
        var state = State.InitialNoDelta; int initial = 0, legacy = 0, plain = 0, fused = 0;
        void Normalize(bool post)
        {
            switch (Q35CudaEngine.PlanNinferResidualNormalization(state, true, post))
            { case Route.LegacyInitial: initial++; break; case Route.LegacyDelta: legacy++; break; case Route.ReadOnlyCanonical: plain++; break; }
        }
        for (int layer = 0; layer < 64; layer++)
        {
            Normalize(false);
            bool outputActuallyFused = layer is not (7 or 31 or 55); // Three BF16 output fallbacks.
            state = Q35CudaEngine.NinferProjectedResidualState(outputActuallyFused); if (outputActuallyFused) fused++;
            Normalize(true);
            state = Q35CudaEngine.NinferProjectedResidualState(true); fused++;
        }
        Normalize(false); // Verify's final norm, with the original output weight.
        Assert.Equal(State.FusedCanonical, state); Assert.Equal(125, fused);
        Assert.Equal(125, plain); Assert.Equal(1, initial); Assert.Equal(3, legacy);
    }

    [Fact]
    public void GraphIdentityIncludesActualModuleAndPolicyWithoutAliasingLegacyGraphs()
    {
        string a = Q35CudaEngine.BuildNinferResidualNormStateSuffix(new string('A', 64));
        string b = Q35CudaEngine.BuildNinferResidualNormStateSuffix(new string('b', 64));
        var legacy = Q35CudaEngine.PlanNinferDecodeGraphKey(1, 8, true, 0, new(0, 42));
        var candidate = Q35CudaEngine.PlanNinferDecodeGraphKey(1, 8, true, 0, new(0, 42), a);
        Assert.NotEqual(legacy, candidate);
        Assert.NotEqual(candidate, Q35CudaEngine.PlanNinferDecodeGraphKey(1, 8, true, 0, new(0, 42), b));
        Assert.Equal(candidate, Q35CudaEngine.PlanNinferDecodeGraphKey(1, 8, true, 0, new(0, 42), a));
        Assert.Contains(Q35Kernels.NinferResidualNormPolicy, a); Assert.EndsWith(new string('a', 64), a);
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.BuildNinferResidualNormStateSuffix("bad"));
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.PlanNinferDecodeGraphKey(2, 8, true, 0, new(0, 42), a));
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.PlanNinferDecodeGraphKey(1, 8, true, 1, new(0, 42), a));
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.PlanNinferDecodeGraphKey(1, 8, true, 0, new(0, 42), ""));
    }

    [Theory]
    [InlineData(false, false)] [InlineData(true, false)] [InlineData(false, true)] [InlineData(true, true)]
    public void ConfigurationIsOnlyPermittedBeforeAnyGemmOrPriorConfiguration(bool configured, bool work)
    {
        if (configured || work) Assert.Throws<InvalidOperationException>(() => Q35Kernels.ValidateNinferResidualNormConfiguration(configured, work));
        else Q35Kernels.ValidateNinferResidualNormConfiguration(configured, work);
    }
}
