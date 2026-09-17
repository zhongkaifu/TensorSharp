// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Reflection;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

public class Qwen35TokenDecodeContractTests
{
    [Theory]
    [InlineData(BackendType.GgmlMetal, false, "1", true)]
    [InlineData(BackendType.GgmlMetal, false, "0", false)]
    [InlineData(BackendType.GgmlMetal, false, null, true)]
    [InlineData(BackendType.GgmlMetal, false, "", true)]
    [InlineData(BackendType.GgmlMetal, true, "1", false)]
    [InlineData(BackendType.GgmlMetal, true, null, false)]
    [InlineData(BackendType.GgmlCuda, false, "1", false)]
    [InlineData(BackendType.GgmlCuda, false, null, false)]
    [InlineData(BackendType.GgmlVulkan, false, "1", false)]
    public void MetalGdnInplaceStateGate_DefaultsOnOnlyForSingleDeviceMetal(
        BackendType backend,
        bool isTensorParallel,
        string? environmentValue,
        bool expected)
    {
        Assert.Equal(
            expected,
            Qwen35Model.ShouldUseMetalGdnInplaceState(
                backend,
                isTensorParallel,
                environmentValue));
    }

    /// <summary>
    /// Device-resident GDN state is for SINGLE-TOKEN calls only.
    ///
    /// <para>A resident call updates the recurrent state in place, so the state
    /// the call started from no longer exists afterwards. That is harmless for a
    /// plain decode step, which is never rolled back, and wrong for a multi-row
    /// verify, whose whole purpose is to be rolled back when a draft is rejected:
    /// SpecSnapshotRecurrentState takes no copy when the state is device-live
    /// because "a verify only reads the live slices", which an in-place update
    /// makes false. Allowing it on verifies made the emitted stream diverge from
    /// plain greedy at token 53 of a measured run.</para>
    ///
    /// <para>Metal stays out entirely: its decode may bind the GDN result by the
    /// backing-base pointer while a resident verify graph retains the offset
    /// state-view pointer.</para>
    /// </summary>
    [Theory]
    [InlineData(BackendType.GgmlMetal, true, -1, 4, false)]
    [InlineData(BackendType.GgmlMetal, true, 4, 4, false)]
    [InlineData(BackendType.GgmlMetal, true, 1, 1, false)]
    // Multi-row verify: never resident, whatever the logit-row shape.
    [InlineData(BackendType.GgmlCuda, true, -1, 4, false)]
    [InlineData(BackendType.GgmlCuda, true, 4, 4, false)]
    [InlineData(BackendType.GgmlCuda, true, 1, 4, false)]
    // Single-token plain/decode steps: resident, which is where the time is.
    [InlineData(BackendType.GgmlCuda, true, 1, 1, true)]
    [InlineData(BackendType.GgmlCuda, true, -1, 1, true)]
    [InlineData(BackendType.GgmlCuda, false, 1, 1, false)]
    public void VerifyResidentState_IsSingleTokenOnlyAndNeverMetal(
        BackendType backend,
        bool residentEnabled,
        int nLogitRows,
        int seqLen,
        bool expected)
    {
        Assert.Equal(
            expected,
            Qwen35Model.ShouldUseVerifyResidentState(
                backend,
                residentEnabled,
                nLogitRows,
                seqLen));
    }

    [Fact]
    public void Bonsai27BMetalPrefillChunk_DefaultIsNarrowlyScopedToExactGeometry()
    {
        Assert.True(Qwen35Model.ShouldUseBonsai27BMetalPrefillChunk(
            BackendType.GgmlMetal,
            numLayers: 64,
            hiddenSize: 5120,
            numHeads: 24,
            numKvHeads: 4,
            headKDim: 128,
            headVDim: 128,
            numKHeads: 16,
            numVHeads: 48));

        Assert.False(Qwen35Model.ShouldUseBonsai27BMetalPrefillChunk(
            BackendType.GgmlCuda,
            numLayers: 64,
            hiddenSize: 5120,
            numHeads: 24,
            numKvHeads: 4,
            headKDim: 128,
            headVDim: 128,
            numKHeads: 16,
            numVHeads: 48));

        Assert.False(Qwen35Model.ShouldUseBonsai27BMetalPrefillChunk(
            BackendType.GgmlMetal,
            numLayers: 63,
            hiddenSize: 5120,
            numHeads: 24,
            numKvHeads: 4,
            headKDim: 128,
            headVDim: 128,
            numKHeads: 16,
            numVHeads: 48));
    }

    [Theory]
    [InlineData(512, 512, 512)]
    [InlineData(513, 512, 511)]
    [InlineData(514, 512, 512)]
    [InlineData(1025, 512, 512)]
    [InlineData(2, 512, 2)]
    [InlineData(1, 512, 1)]
    [InlineData(2, 1, 1)]
    public void RefillChunkLength_NeverStrandsFinalToken(
        int remaining,
        int chunkSize,
        int expected)
    {
        Assert.Equal(expected, Qwen35Model.ComputeRefillChunkLength(remaining, chunkSize));
    }

    [Theory]
    [InlineData(0, 512)]
    [InlineData(512, 0)]
    public void RefillChunkLength_RejectsInvalidInputs(int remaining, int chunkSize)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Qwen35Model.ComputeRefillChunkLength(remaining, chunkSize));
    }

    [Fact]
    public void HostPrefillFallback_DrainsDeviceStateBeforeReadingHostMirrors()
    {
        var calls = new List<string>();

        Qwen35Model.PrepareHostPrefillFallback(
            () => calls.Add("drain-device-state"),
            () => calls.Add("sync-kv"),
            () => calls.Add("sync-fused-decode-state"),
            () => calls.Add("invalidate-decode-bindings"));

        Assert.Equal(
            new[]
            {
                "drain-device-state",
                "sync-kv",
                "sync-fused-decode-state",
                "invalidate-decode-bindings",
            },
            calls);
    }

    [Fact]
    public void HostPrefillFallback_DrainFailureDoesNotEnterHostPath()
    {
        var calls = new List<string>();

        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() =>
            Qwen35Model.PrepareHostPrefillFallback(
                () => throw new InvalidOperationException("device state unavailable"),
                () => calls.Add("sync-kv"),
                () => calls.Add("sync-fused-decode-state"),
                () => calls.Add("invalidate-decode-bindings")));

        Assert.Equal("device state unavailable", error.Message);
        Assert.Empty(calls);
    }

    [Theory]
    [InlineData(
        "Qwen35Model.Speculative.cs",
        "if (TryFusedVerifyTrunk(hidden, startPos, seqLen, hAllOut, logitsOut, allLogitsRows))")]
    [InlineData(
        "Qwen35Model.DFlash.cs",
        "private unsafe void DFlashSpecForwardPerOp(")]
    public void FailedVerifyPerOpRoutes_CrossTheSharedHostFallbackBarrier(
        string fileName,
        string routeStartMarker)
    {
        string source = File.ReadAllText(Path.Combine(
            FindRepositoryRoot(),
            "TensorSharp.Models",
            "Models",
            "Qwen35",
            fileName));

        int routeStart = source.IndexOf(routeStartMarker, StringComparison.Ordinal);
        Assert.True(routeStart >= 0, $"Fallback route not found in {fileName}.");
        int layerLoop = source.IndexOf(
            "for (int layer = 0; layer < Config.NumLayers; layer++)",
            routeStart,
            StringComparison.Ordinal);
        Assert.True(layerLoop > routeStart, $"Per-op layer loop not found in {fileName}.");
        string transition = source[routeStart..layerLoop];

        Assert.Contains("PrepareHostPrefillFallback();", transition);
        Assert.DoesNotContain("EnsureKvCacheHostSynchronized();", transition);
        Assert.DoesNotContain("EnsureFusedDecodeStateHostSynchronized();", transition);
    }

    [Fact]
    public void VerifyDispose_DiscardsManagedStateBeforeNativeReset()
    {
        var calls = new List<string>();

        Qwen35Model.DiscardVerifyStateForDispose(
            () => calls.Add("discard-managed-state"),
            () => calls.Add("reset-native-state"));

        Assert.Equal(
            new[] { "discard-managed-state", "reset-native-state" },
            calls);
    }

    [Fact]
    public void VerifyDispose_NativeResetFailureStillLeavesStateDiscarded()
    {
        bool managedStateDiscarded = false;

        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() =>
            Qwen35Model.DiscardVerifyStateForDispose(
                () => managedStateDiscarded = true,
                () => throw new InvalidOperationException("native reset failed")));

        Assert.True(managedStateDiscarded);
        Assert.Equal("native reset failed", error.Message);
    }

    [Fact]
    public void NativeDecodePools_AreLockedAcrossUseAndTeardown()
    {
        string source = File.ReadAllText(Path.Combine(
            FindRepositoryRoot(),
            "TensorSharp.GGML.Native",
            "ggml_ops_qwen35_decode.cpp")).ReplaceLineEndings("\n");
        const string lockLine =
            "std::lock_guard<std::recursive_mutex> lock(q35_decode_mutex());";

        string Slice(string startMarker, string endMarker)
        {
            int start = source.IndexOf(startMarker, StringComparison.Ordinal);
            Assert.True(start >= 0, $"Native marker not found: {startMarker}");
            int end = source.IndexOf(endMarker, start + startMarker.Length, StringComparison.Ordinal);
            Assert.True(end > start, $"Native marker not found after {startMarker}: {endMarker}");
            return source[start..end];
        }

        Assert.Contains("std::recursive_mutex& q35_decode_mutex()", source);

        string solo = Slice(
            "int qwen35_model_decode_impl(",
            "TSG_EXPORT int TSGgml_Qwen35ModelDecode(");
        int arenaTouch = solo.IndexOf("tsg_q35arena::on_external_touch", StringComparison.Ordinal);
        int soloLock = solo.IndexOf(lockLine, StringComparison.Ordinal);
        int soloPool = solo.IndexOf("g_q35dc_pool.find", StringComparison.Ordinal);
        Assert.True(arenaTouch >= 0 && soloLock > arenaTouch && soloPool > soloLock,
            "Solo decode must preserve arena->pool lock ordering and lock before its first retained-pool access.");

        string attention = Slice(
            "TSG_EXPORT int TSGgml_Qwen35AttentionLayerDecode(",
            "// ============================================================================\n// Qwen3.5/3.6 FULL-MODEL decode");
        Assert.Contains(lockLine, attention);

        string dropAndReset = Slice(
            "void tsg_q35_drop_decode_graphs_for_kv(",
            "// ============================================================================\n// TSGgml_Qwen35ModelDecodeBatched");
        Assert.Equal(2, dropAndReset.Split(lockLine, StringSplitOptions.None).Length - 1);

        string batched = source[source.IndexOf(
            "TSG_EXPORT int TSGgml_Qwen35ModelDecodeBatched(",
            StringComparison.Ordinal)..];
        int batchedLock = batched.IndexOf(lockLine, StringComparison.Ordinal);
        int batchedPool = batched.IndexOf("qwen35_model_decode_batched_impl(", StringComparison.Ordinal);
        int resetStart = batched.IndexOf(
            "TSG_EXPORT void TSGgml_Qwen35ResetBatchedDecodeCache()",
            StringComparison.Ordinal);
        int resetLock = batched.IndexOf(lockLine, resetStart, StringComparison.Ordinal);
        int resetPool = batched.IndexOf("g_q35bdc.reset();", resetStart, StringComparison.Ordinal);
        Assert.True(batchedLock >= 0 && batchedPool > batchedLock,
            "Batched decode must lock before entering the retained-pool implementation.");
        Assert.True(resetStart >= 0 && resetLock > resetStart && resetPool > resetLock,
            "Batched reset must lock before freeing the retained graph.");
    }

    [Fact]
    public void MetalGdnInplaceStateLayout_UsesOneAttentionRowAsBackingPrefix()
    {
        const int numVHeads = 3;
        const int headVDim = 5;
        const int headKDim = 7;
        const long attentionElements = numVHeads * headVDim;
        const long stateElements = attentionElements * headKDim;
        var allocator = new CpuAllocator(BlasEnum.DotNet);

        using Tensor ordinary = Qwen35Model.AllocateGdnDeltaStateTensor(
            allocator,
            useMetalInplaceLayout: false,
            numVHeads,
            headVDim,
            headKDim);
        Assert.Equal(0, ordinary.StorageOffset);
        Assert.Equal(stateElements, ordinary.Storage.ElementCount);

        using Tensor inplace = Qwen35Model.AllocateGdnDeltaStateTensor(
            allocator,
            useMetalInplaceLayout: true,
            numVHeads,
            headVDim,
            headKDim);
        Assert.Equal(attentionElements, inplace.StorageOffset);
        Assert.Equal(attentionElements + stateElements, inplace.Storage.ElementCount);
        Assert.Equal(stateElements * sizeof(float), Qwen35Model.GdnDeltaStateBytes(inplace));
        Assert.Equal(
            attentionElements * sizeof(float),
            TensorComputePrimitives.GetStoragePointer(inplace).ToInt64() -
                TensorComputePrimitives.GetStorageBasePointer(inplace).ToInt64());
    }

    [Fact]
    public void DirectTokenDecodeRoute_AcceptsOrdinarySingleTokenMetalDecode()
    {
        Assert.True(Qwen35Model.CanTryDirectTokenDecode(
            BackendType.GgmlMetal,
            isTensorParallel: false,
            sequenceLength: 1,
            hasVisionEmbeddings: false,
            hasMRoPEPositions: false,
            tokenId: 42,
            vocabSize: 128));
    }

    [Theory]
    [InlineData(BackendType.GgmlCuda, false, 1, false, false, 42, 128)]
    [InlineData(BackendType.GgmlVulkan, false, 1, false, false, 42, 128)]
    [InlineData(BackendType.GgmlMetal, true, 1, false, false, 42, 128)]
    [InlineData(BackendType.GgmlMetal, false, 2, false, false, 42, 128)]
    [InlineData(BackendType.GgmlMetal, false, 1, true, false, 42, 128)]
    [InlineData(BackendType.GgmlMetal, false, 1, false, true, 42, 128)]
    [InlineData(BackendType.GgmlMetal, false, 1, false, false, -1, 128)]
    [InlineData(BackendType.GgmlMetal, false, 1, false, false, 128, 128)]
    [InlineData(BackendType.GgmlMetal, false, 1, false, false, 0, 0)]
    public void DirectTokenDecodeRoute_RejectsUnsupportedOrInvalidInputs(
        BackendType backend,
        bool isTensorParallel,
        int sequenceLength,
        bool hasVisionEmbeddings,
        bool hasMRoPEPositions,
        int tokenId,
        int vocabSize)
    {
        Assert.False(Qwen35Model.CanTryDirectTokenDecode(
            backend,
            isTensorParallel,
            sequenceLength,
            hasVisionEmbeddings,
            hasMRoPEPositions,
            tokenId,
            vocabSize));
    }

    [Fact]
    public void ArenaGraphFailure_FailsClosedInsteadOfAllowingSerialFallback()
    {
        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() =>
            Qwen35Model.ThrowIfArenaDecodeStateUnrecoverable(
                status: -1,
                nativeError: "Qwen3.5 arena batched decode: graph execution failed."));

        Assert.Contains("partially advanced recurrent state", error.Message);
        Assert.Contains("serial fallback would use stale host state", error.Message);

        // Zero is a safe pre-compute shape/capability decline; success is one.
        Qwen35Model.ThrowIfArenaDecodeStateUnrecoverable(0, "safe decline");
        Qwen35Model.ThrowIfArenaDecodeStateUnrecoverable(1, string.Empty);
    }

    [Theory]
    [InlineData(nameof(GgmlBasicOps.Qwen35ModelDecode), "TSGgml_Qwen35ModelDecode")]
    [InlineData(nameof(GgmlBasicOps.Qwen35ModelDecodeToken), "TSGgml_Qwen35ModelDecodeToken")]
    public void DecodePInvoke_MatchesPublicManagedParameterContract(
        string publicMethodName,
        string nativeMethodName)
    {
        MethodInfo publicMethod = typeof(GgmlBasicOps).GetMethod(
            publicMethodName,
            BindingFlags.Public | BindingFlags.Static)!;

        Type nativeType = typeof(GgmlBasicOps).Assembly.GetType(
            "TensorSharp.GGML.GgmlNative",
            throwOnError: true)!;
        MethodInfo nativeMethod = nativeType.GetMethod(
            nativeMethodName,
            BindingFlags.NonPublic | BindingFlags.Static)!;

        Assert.NotNull(publicMethod);
        Assert.NotNull(nativeMethod);
        Assert.Equal(typeof(bool), publicMethod.ReturnType);
        Assert.Equal(typeof(int), nativeMethod.ReturnType);
        Assert.Equal(
            publicMethod.GetParameters().Select(parameter => parameter.ParameterType),
            nativeMethod.GetParameters().Select(parameter => parameter.ParameterType));

        LibraryImportAttribute import = nativeMethod.GetCustomAttribute<LibraryImportAttribute>()!;
        Assert.NotNull(import);
        Assert.Equal("GgmlOps", import.LibraryName);
        UnmanagedCallConvAttribute callConv = nativeMethod.GetCustomAttribute<UnmanagedCallConvAttribute>()!;
        Assert.NotNull(callConv);
        Assert.Contains(typeof(System.Runtime.CompilerServices.CallConvCdecl), callConv.CallConvs!);
        Assert.Equal(nativeMethodName, nativeMethod.Name);

        ParameterInfo reseedParameter = nativeMethod.GetParameters()[2];
        Assert.Equal(typeof(bool), reseedParameter.ParameterType);
        MarshalAsAttribute marshalAs = reseedParameter.GetCustomAttribute<MarshalAsAttribute>()!;
        Assert.NotNull(marshalAs);
        Assert.Equal(UnmanagedType.Bool, marshalAs.Value);
    }

    [Fact]
    public void Qwen35LayerDescriptor_HasExpectedBlittableLayout()
    {
        FieldInfo[] fields = typeof(Qwen35LayerDecodeArgs).GetFields(
            BindingFlags.Public | BindingFlags.Instance);

        // Every addition is appended at the END of its own run, so the struct stays
        // a pointers / int64 / int32 sequence and the native TSGgmlQwen35LayerDesc
        // keeps the same offsets for every field before it. The two most recent:
        //   CpuMoe                    per-layer MoE CPU offload (--n-cpu-moe)
        //   FfnGateW / FfnUpW (+ their shapes and types)
        //                             the dense FFN of a mixed-quant "UD" layer whose
        //                             ffn_gate and ffn_up have different GGML types
        //                             and so cannot be fused into one tensor
        Assert.Equal(36, fields.Count(field => field.FieldType == typeof(IntPtr)));
        Assert.Equal(54, fields.Count(field => field.FieldType == typeof(long)));
        Assert.Equal(26, fields.Count(field => field.FieldType == typeof(int)));
        Assert.All(fields, field => Assert.True(
            field.FieldType == typeof(IntPtr) ||
            field.FieldType == typeof(long) ||
            field.FieldType == typeof(int),
            $"Unexpected non-blittable field {field.Name}: {field.FieldType}."));

        static long Align(long value, int alignment)
            => (value + alignment - 1) / alignment * alignment;

        long int64Start = Align(36L * IntPtr.Size, sizeof(long));
        long int32Start = int64Start + 54L * sizeof(long);
        long expectedSize = Align(
            int32Start + 26L * sizeof(int),
            Math.Max(IntPtr.Size, sizeof(long)));

        Assert.Equal(0, Marshal.OffsetOf<Qwen35LayerDecodeArgs>(
            nameof(Qwen35LayerDecodeArgs.AttnNormW)).ToInt64());
        Assert.Equal(int64Start, Marshal.OffsetOf<Qwen35LayerDecodeArgs>(
            nameof(Qwen35LayerDecodeArgs.QkvNe0)).ToInt64());
        Assert.Equal(int32Start, Marshal.OffsetOf<Qwen35LayerDecodeArgs>(
            nameof(Qwen35LayerDecodeArgs.StructBytes)).ToInt64());
        Assert.Equal(expectedSize, Marshal.SizeOf<Qwen35LayerDecodeArgs>());

        // The int32 run's ORDER is what the struct_bytes handshake cannot check -
        // it only catches a size change - so pin the tail explicitly. CpuMoe kept
        // its slot when the split-gate/up pair was appended after it.
        Assert.Equal(
            int32Start + 23L * sizeof(int),
            Marshal.OffsetOf<Qwen35LayerDecodeArgs>(nameof(Qwen35LayerDecodeArgs.CpuMoe)).ToInt64());
        Assert.Equal(
            int32Start + 24L * sizeof(int),
            Marshal.OffsetOf<Qwen35LayerDecodeArgs>(nameof(Qwen35LayerDecodeArgs.FfnGateType)).ToInt64());
        Assert.Equal(
            int32Start + 25L * sizeof(int),
            Marshal.OffsetOf<Qwen35LayerDecodeArgs>(nameof(Qwen35LayerDecodeArgs.FfnUpType)).ToInt64());
        // Same for the pointer and int64 runs.
        Assert.Equal(
            34L * IntPtr.Size,
            Marshal.OffsetOf<Qwen35LayerDecodeArgs>(nameof(Qwen35LayerDecodeArgs.FfnUpW)).ToInt64());
        Assert.Equal(
            35L * IntPtr.Size,
            Marshal.OffsetOf<Qwen35LayerDecodeArgs>(nameof(Qwen35LayerDecodeArgs.ProjScales)).ToInt64());
        Assert.Equal(
            int64Start + 48L * sizeof(long),
            Marshal.OffsetOf<Qwen35LayerDecodeArgs>(nameof(Qwen35LayerDecodeArgs.FfnGateNe0)).ToInt64());
    }

    /// <summary>
    /// Every fused Qwen3.5 graph rotates at the RoPE position, not the KV index: the
    /// solo decode adds rope_pos_delta, the verify adds it to its scalar positions,
    /// and the arena reads rope_positions. A sequence past an image has a non-zero
    /// M-RoPE delta, so a site that went back to the KV index would silently decode
    /// at the pre-fix positions (Qwen35ImageFollowUpExactnessTests is the model-gated
    /// proof; this is the portable guard).
    /// </summary>
    [Fact]
    public void NativeFusedGraphs_RotateAtTheRopePositionNotTheKvIndex()
    {
        string Read(string file) => File.ReadAllText(Path.Combine(
            FindRepositoryRoot(), "TensorSharp.GGML.Native", file)).ReplaceLineEndings("\n");

        string decode = Read("ggml_ops_qwen35_decode.cpp");
        // The whole-model decode (the per-layer TSGgml_Qwen35AttentionLayerDecode still
        // rotates at its KV index; the managed caller refuses it past an image).
        int implStart = decode.IndexOf("int qwen35_model_decode_impl(", StringComparison.Ordinal);
        int implEnd = decode.IndexOf("TSG_EXPORT int TSGgml_Qwen35ModelDecode(", StringComparison.Ordinal);
        Assert.True(implStart >= 0 && implEnd > implStart);
        string solo = decode[implStart..implEnd];
        Assert.DoesNotContain("pos_val = position;", solo);
        Assert.Equal(2, solo.Split("std::int32_t pos_val = position + rope_pos_delta;").Length - 1);
        Assert.Contains("TSG_EXPORT int TSGgml_Qwen35RopePositionAbi()", decode);

        string verify = Read("ggml_ops_qwen35_verify.cpp");
        Assert.DoesNotContain("pv[i] = start_pos + i;", verify);
        Assert.DoesNotContain("pos_vals[i] = start_pos + i;", verify);
        Assert.Contains("pv[i] = start_pos + rope_pos_delta + i;", verify);
        Assert.Contains("pos_vals[i] = start_pos + rope_pos_delta + i;", verify);

        string arena = Read("ggml_ops_qwen35_batched_arena.cpp");
        Assert.DoesNotContain("e.pos_stage[s] = positions[i];", arena);
        Assert.Contains("e.pos_stage[s] = rope_positions != nullptr ? rope_positions[i] : positions[i];", arena);
    }

    [Fact]
    public void Qwen35_ReusesAcrossMedia_AndCheckpointFilesCarryTheDelta()
    {
        var model = (Qwen35Model)System.Runtime.CompilerServices.RuntimeHelpers.GetUninitializedObject(typeof(Qwen35Model));
        Assert.True(model.SupportsReuseAcrossMediaSpan);
        // Version 2 adds the M-RoPE delta; a version-1 file is refused on import.
        Assert.Equal(2, Qwen35Model.CheckpointFileVersion);
    }

    private static string FindRepositoryRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "TensorSharp.sln"))
                || File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx")))
                return dir.FullName;
            dir = dir.Parent;
        }
        throw new DirectoryNotFoundException("Could not locate the TensorSharp repository root.");
    }
}
