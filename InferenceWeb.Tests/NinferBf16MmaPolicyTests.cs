// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;

namespace InferenceWeb.Tests;

/// <summary>CPU admission only. Real entry-name witnesses come from the qualified
/// R52 NVCC image dcd7283a, rather than being reconstructed by the name builder.</summary>
public sealed class NinferBf16MmaPolicyTests
{
    private static bool Plan(string? option, bool qualified = true, int sms = 170, int context = 40960,
        int rows = 2048, bool verify = true, bool graphs = true, bool fi = false, bool exact = false)
        => Q35CudaEngine.PlanNinferBf16Mma(true, qualified, sms, context, rows, verify, graphs, fi, exact, option!);

    [Theory]
    [InlineData(null, true)] [InlineData("0", false)] [InlineData("1", true)]
    public void QualifiedProfileSelectsAutomaticallyWithExplicitOptOut(string? value, bool expected) => Assert.Equal(expected, Plan(value));

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
    [InlineData("")] [InlineData(" ")] [InlineData("true")] [InlineData("01")] [InlineData("2")] [InlineData(" 1")]
    public void MalformedNinferOptionCannotHideBehindAnUnsupportedProfile(string value)
        => Assert.Throws<ArgumentException>(() => Plan(value, qualified: false));

    [Fact]
    public void RequiredCandidateRejectsUnmeasuredProfilesAndProjectionOverrides()
    {
        Action[] requests = { () => Plan("1", qualified: false), () => Plan("1", sms: 169), () => Plan("1", sms: 171),
            () => Plan("1", context: 40959), () => Plan("1", context: 65535), () => Plan("1", rows: 7),
            () => Plan("1", verify: false), () => Plan("1", graphs: false), () => Plan("1", fi: true), () => Plan("1", exact: true) };
        foreach (var request in requests) Assert.Throws<NotSupportedException>(request);
        Assert.False(Plan(null, qualified: false, graphs: false, fi: true));
        Assert.False(Plan("0", qualified: false, graphs: false, exact: true));
    }

    [Fact]
    public void AutomaticSelectionDeclinesEveryUnmeasuredProfileAndOverride()
    {
        bool[] rejected = { Plan(null, qualified: false), Plan(null, sms: 169), Plan(null, sms: 171),
            Plan(null, context: 40959), Plan(null, context: 65535), Plan(null, rows: 7),
            Plan(null, verify: false), Plan(null, graphs: false), Plan(null, fi: true), Plan(null, exact: true) };
        Assert.All(rejected, x => Assert.False(x));
    }

    [Fact]
    public void MissingPackagedAssetFallsBackOnlyForAutomaticSelectionBeforeUse()
    {
        var calls = new List<string>();
        string FatbinOnly(string name) { calls.Add(name); return name.EndsWith(".fatbin") ? "/assets/cuda_kernels/tensorsharp_ninfer_bf16_mma.fatbin" : null!; }
        Assert.EndsWith(".fatbin", Q35CudaEngine.ResolveNinferBf16MmaAsset(false, FatbinOnly));
        Assert.Equal(new[] { "tensorsharp_ninfer_bf16_mma.ptx", "tensorsharp_ninfer_bf16_mma.fatbin" }, calls);
        Assert.Null(Q35CudaEngine.ResolveNinferBf16MmaAsset(false, _ => null!));
        Assert.Throws<FileNotFoundException>(() => Q35CudaEngine.ResolveNinferBf16MmaAsset(true, _ => null!));
        Assert.True(Q35CudaEngine.NinferBf16MmaAssetMissing(new FileNotFoundException()));
        Assert.True(Q35CudaEngine.NinferBf16MmaAssetMissing(new DirectoryNotFoundException()));
    }

    [Fact]
    public void InvalidAbiAllocationAndCudaFaultsAreNeverMissingAssets()
    {
        Exception[] failures = { new InvalidOperationException("ABI"), new OutOfMemoryException(), new IOException(),
            new UnauthorizedAccessException(), new AccessViolationException(), new CudaException(2, "allocation"),
            new CudaException(201, "context"), new CudaException(500, "missing kernel entry is an ABI failure"),
            new CudaException(700, "launch"), new CudaException(719, "execution") };
        Assert.All(failures, error => Assert.False(Q35CudaEngine.NinferBf16MmaAssetMissing(error)));
    }

    [Theory]
    [InlineData(null)] [InlineData("0")] [InlineData("1")] [InlineData("invalid")]
    public void GgufDoesNotParseOrEnableTheCandidate(string? value)
        => Assert.False(Q35CudaEngine.PlanNinferBf16Mma(false, false, 0, 0, 0, false, false, false, false, value!));

    private static bool Pass(bool enabled = true, int groups = 1, int window = 8, int lanes = 1,
        bool verify = true, bool counts = false, bool embeddings = false, bool taps = false, bool drafter = false, int first = 0)
        => Q35CudaEngine.UseNinferBf16MmaPass(enabled, groups, window, lanes, verify, counts, embeddings, taps, drafter, first);

    [Fact]
    public void EightPhysicalRowsAreInsufficientWithoutOneCompleteVerifyLane()
    {
        Assert.True(Pass());
        bool[] rejected = { Pass(enabled: false), Pass(groups: 2), Pass(window: 4, lanes: 2), Pass(window: 1, lanes: 8),
            Pass(window: 7), Pass(lanes: 2), Pass(verify: false), Pass(counts: true), Pass(embeddings: true),
            Pass(taps: true), Pass(drafter: true), Pass(first: 1) };
        Assert.All(rejected, x => Assert.False(x));
    }

    [Theory]
    [InlineData(14336, 5120, 448, 81920, 146800640, 458752)]
    [InlineData(5120, 6144, 160, 98304, 62914560, 163840)]
    public void ProjectionStorageIsBf16InputAndWeightsWithUnroundedF32Output(int n, int k, int blocks,
        long input, long weights, long output)
    {
        var plan = CudaNinferBf16MmaModule.Plan(n, k, 8);
        Assert.Equal(blocks, plan.Blocks); Assert.Equal(input, plan.InputBytes);
        Assert.Equal(weights, plan.WeightBytes); Assert.Equal(output, plan.OutputBytes);
        Assert.True(Q35CudaEngine.NinferBf16MmaProjectionEligible(true, CudaNinferProjectionFormat.BFloat16, 8, n, k));
        Assert.False(Q35CudaEngine.NinferBf16MmaProjectionEligible(false, CudaNinferProjectionFormat.BFloat16, 8, n, k));
        Assert.False(Q35CudaEngine.NinferBf16MmaProjectionEligible(true, CudaNinferProjectionFormat.NvFp4Planes, 8, n, k));
        Assert.False(Q35CudaEngine.NinferBf16MmaProjectionEligible(true, CudaNinferProjectionFormat.BFloat16, 7, n, k));
        Assert.Throws<ArgumentOutOfRangeException>(() => CudaNinferBf16MmaModule.Plan(n, k, 9));
        Assert.True(plan.InputBytes <= CudaNinferBf16MmaModule.ScratchBytes);
    }

    [Fact]
    public void AlphaBetaAndOtherBf16ShapesStayOnTheirExistingPath()
    {
        Assert.False(Q35CudaEngine.NinferBf16MmaProjectionEligible(true, CudaNinferProjectionFormat.BFloat16, 8, 96, 5120));
        Assert.Throws<ArgumentOutOfRangeException>(() => CudaNinferBf16MmaModule.Plan(96, 5120, 8));
        Assert.Throws<ArgumentOutOfRangeException>(() => CudaNinferBf16MmaModule.Plan(5120, 17408, 8));
    }

    [Fact]
    public void OutputMustBeACompleteNonoverlappingF32Span()
    {
        var p = CudaNinferBf16MmaModule.Plan(14336, 5120, 8);
        IntPtr input = (IntPtr)0x20000000, weights = (IntPtr)0x10000000, output = (IntPtr)0x30000000;
        CudaNinferBf16MmaModule.ValidateBuffers(input, weights, output, p);
        CudaNinferBf16MmaModule.ValidateBuffers(input, weights, weights + (nint)p.WeightBytes, p);
        Assert.Throws<ArgumentException>(() => CudaNinferBf16MmaModule.ValidateBuffers(input, weights, input + 4, p));
        Assert.Throws<ArgumentException>(() => CudaNinferBf16MmaModule.ValidateBuffers(input, weights, weights + (nint)p.WeightBytes - 4, p));
        Assert.Throws<ArgumentException>(() => CudaNinferBf16MmaModule.ValidateBuffers(IntPtr.Zero, weights, output, p));
        Assert.Throws<ArgumentException>(() => CudaNinferBf16MmaModule.ValidateBuffers(input + 2, weights, output, p));
        Assert.Throws<ArgumentException>(() => CudaNinferBf16MmaModule.ValidateBuffers((IntPtr)(long.MaxValue - 15), weights, output, p));
    }

    [Fact]
    public void GraphAndPrefixIdentitiesBindTheActualF32PolicyAndModule()
    {
        string identity = Q35CudaEngine.BuildNinferBf16MmaStateSuffix(new string('A', 64));
        string another = Q35CudaEngine.BuildNinferBf16MmaStateSuffix(new string('b', 64));
        var old = Q35CudaEngine.PlanNinferDecodeGraphKey(1, 8, true, 0, new(0, 42));
        var next = Q35CudaEngine.PlanNinferDecodeGraphKey(1, 8, true, 0, new(0, 42), bf16MmaIdentity: identity);
        Assert.NotEqual(old, next);
        Assert.NotEqual(next, Q35CudaEngine.PlanNinferDecodeGraphKey(1, 8, true, 0, new(0, 42), bf16MmaIdentity: another));
        Assert.Equal(next, Q35CudaEngine.PlanNinferDecodeGraphKey(1, 8, true, 0, new(0, 42), bf16MmaIdentity: identity));
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.PlanNinferDecodeGraphKey(2, 4, true, 0, new(0, 42), bf16MmaIdentity: identity));
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.PlanNinferDecodeGraphKey(1, 8, false, 0, new(0, 42), bf16MmaIdentity: identity));
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.PlanNinferDecodeGraphKey(1, 8, true, 1, new(0, 42), bf16MmaIdentity: identity));
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.PlanNinferDecodeGraphKey(1, 8, true, 0, new(0, 42), bf16MmaIdentity: identity.ToUpperInvariant()));
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.BuildNinferBf16MmaStateSuffix("bad"));
        Assert.Contains(CudaNinferBf16MmaModule.ArithmeticPolicy, identity); Assert.EndsWith(new string('a', 64), identity);
    }

    public static IEnumerable<object[]> ActualNativeEntries()
    {
        yield return new object[] { 5120, 6144, false, "_ZN6ninfer3ops6detail20bf16_gemm_mma_kernelINS1_16Bf16GemvGeometryILi5120ELi6144EEENS1_15Bf16MmaScheduleILi32ELi32ELi256ELi16ELi8ELi3ELi1ELNS0_5CacheE1ELS6_1ELNS1_23Bf16MmaFragmentPipelineE1ELNS1_13Bf16MmaRasterE0ELNS1_14Bf16MmaSwizzleE1ELi1EEELb0EN11tensorsharp15ninfer_bf16_mma24TsNinferBf16MmaF32OutputILi5120EEEEEvPK13__nv_bfloat16SH_T2_i" };
        yield return new object[] { 5120, 6144, true, "_ZN6ninfer3ops6detail20bf16_gemm_mma_kernelINS1_16Bf16GemvGeometryILi5120ELi6144EEENS1_15Bf16MmaScheduleILi32ELi32ELi256ELi16ELi8ELi3ELi1ELNS0_5CacheE1ELS6_1ELNS1_23Bf16MmaFragmentPipelineE1ELNS1_13Bf16MmaRasterE0ELNS1_14Bf16MmaSwizzleE1ELi1EEELb0ENS1_23Bf16MmaContiguousOutputEEEvPK13__nv_bfloat16SE_T2_i" };
        yield return new object[] { 14336, 5120, false, "_ZN6ninfer3ops6detail20bf16_gemm_mma_kernelINS1_16Bf16GemvGeometryILi14336ELi5120EEENS1_15Bf16MmaScheduleILi32ELi32ELi256ELi16ELi8ELi3ELi1ELNS0_5CacheE1ELS6_1ELNS1_23Bf16MmaFragmentPipelineE1ELNS1_13Bf16MmaRasterE0ELNS1_14Bf16MmaSwizzleE1ELi1EEELb0EN11tensorsharp15ninfer_bf16_mma24TsNinferBf16MmaF32OutputILi14336EEEEEvPK13__nv_bfloat16SH_T2_i" };
        yield return new object[] { 14336, 5120, true, "_ZN6ninfer3ops6detail20bf16_gemm_mma_kernelINS1_16Bf16GemvGeometryILi14336ELi5120EEENS1_15Bf16MmaScheduleILi32ELi32ELi256ELi16ELi8ELi3ELi1ELNS0_5CacheE1ELS6_1ELNS1_23Bf16MmaFragmentPipelineE1ELNS1_13Bf16MmaRasterE0ELNS1_14Bf16MmaSwizzleE1ELi1EEELb0ENS1_23Bf16MmaContiguousOutputEEEvPK13__nv_bfloat16SE_T2_i" };
    }
    [Theory]
    [MemberData(nameof(ActualNativeEntries))]
    public void BindingsMatchAllFourActualQualifiedNvccEntryNames(int n, int k, bool stock, string actual)
        => Assert.Equal(actual, CudaNinferBf16MmaModule.KernelName(n, k, stock));
    [Theory]
    [InlineData(false, 24)] [InlineData(true, 32)]
    public void TokensFollowTheWholeByValueOutputPolicy(bool stock, ulong tokensOffset)
    {
        CudaNinferBf16MmaModule.ValidateParameterStorage(3, stock, tokensOffset, 4);
        Assert.Throws<InvalidOperationException>(() => CudaNinferBf16MmaModule.ValidateParameterStorage(3, stock, tokensOffset - 8, 4));
        Assert.Throws<InvalidOperationException>(() => CudaNinferBf16MmaModule.ValidateParameterStorage(3, stock, tokensOffset, 8));
        Assert.Throws<ArgumentOutOfRangeException>(() => CudaNinferBf16MmaModule.ValidateParameterStorage(4, stock, tokensOffset + 4, 4));
        Assert.Equal(98304, CudaNinferBf16MmaModule.DynamicSharedBytes);
        Assert.Equal(256, CudaNinferBf16MmaModule.Threads);
    }

}
