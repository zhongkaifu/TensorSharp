// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.IO;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

public sealed unsafe partial class Q35CudaEngine
{
    internal const string NinferAutoA16Environment = "TS_Q35_NINFER_A16_AUTO";
    internal const string NinferAutoRepackEnvironment = "TS_Q35_NINFER_REPACK";
    internal const string NinferQualifiedPrefillAttentionAsset = "tensorsharp_ninfer_prefill_attention.ptx";
    internal const string NinferQualifiedA16Asset = "tensorsharp_q35_ninfer_a16_legacy_exact_vector.ptx";
    internal const string NinferQualifiedRepackAsset = "tensorsharp_ninfer_repack.ptx";
    // Ignis-original prefill kernels, selected automatically for the qualified recipe:
    // the Br64/Bc64 BF16 prompt attention and the chunked GDN recurrence.
    internal const string NinferAutoPrefillAttentionEnvironment = "TS_Q35_NINFER_PREFILL_ATTN";
    internal const string NinferAutoGdnChunkEnvironment = "TS_Q35_NINFER_GDN_CHUNK";
    internal const string NinferQualifiedGdnChunkAsset = "tensorsharp_ninfer_gdn_chunk.ptx";
    private bool _ninferAutomaticPrefillAttention, _ninferAutomaticGdnChunk;
    private readonly string _ninferAutoA16Option = Environment.GetEnvironmentVariable(NinferAutoA16Environment);
    private readonly string _ninferAutoRepackOption = Environment.GetEnvironmentVariable(NinferAutoRepackEnvironment);
    private NinferQualifiedDefaults _ninferQualifiedDefaults;

    /// <summary>How the current engine selected its A16 implementation. The actual
    /// module hash remains part of KVStateNamespaceSuffix only after successful loading.</summary>
    public string NinferA16Selection { get; private set; } = "not-ninfer";
    public string NinferA16ModulePath { get; private set; }
    public string NinferA16ImageSha256 { get; private set; }
    public string NinferLoadRepackSelection { get; private set; } = "not-ninfer";
    public string NinferLoadRepackModulePath { get; private set; }
    public string NinferDecodeGraphsSelection { get; private set; } = "not-ninfer";
    public bool NinferDecodeGraphsEnabled => _ninferDecodeGraphsEnabled;

    internal sealed record NinferQualifiedDefaults(bool Qualified, bool AutoA16, bool AutoRepack,
        bool DecodeGraphs, string GraphSelection);
    private sealed record NinferGraphSetting(string Value, bool ExplicitEnabled);

    private static NinferGraphSetting SnapshotNinferGraphSetting(string value)
        => new(value, NinferDecodeGraphsRequested(value));

    internal static bool NinferAutomaticOptimizationRequested(string value, string option) => value switch
    {
        null or "1" => true,
        "0" => false,
        _ => throw new ArgumentException(option + " accepts only 0 or 1.")
    };

    internal static bool NinferQualifiedDefaultGeometry(bool ninfer, int layers, int hidden, int ff,
        int vocabulary, int heads, int kvHeads, int headDim, int major, int minor)
        => ninfer && layers == 64 && hidden == 5120 && ff == 17408 && vocabulary == 248320
            && heads == 24 && kvHeads == 4 && headDim == 256 && major == 12 && minor == 0;

    internal static NinferQualifiedDefaults PlanNinferQualifiedDefaults(bool qualified,
        string a16Option, string repackOption, string graphsOption)
    {
        bool a16 = NinferAutomaticOptimizationRequested(a16Option, NinferAutoA16Environment);
        bool repack = NinferAutomaticOptimizationRequested(repackOption, NinferAutoRepackEnvironment);
        // Explicit graph choices retain their existing semantics, including on
        // other hardware. An automatic default never expands device coverage.
        bool graphs = graphsOption == null ? qualified : NinferDecodeGraphsRequested(graphsOption);
        return new(qualified, qualified && a16, qualified && repack, graphs,
            graphsOption != null ? (graphs ? "explicit" : "disabled") : qualified ? "auto-sm120" : "unsupported-hardware");
    }

    private void InitializeNinferQualifiedDefaults()
    {
        _ninferDecodeGraphsEnabled = _ninferDecodeGraphsSetting.ExplicitEnabled;
        if (_m.Ninfer == null) return; // New NInfer controls cannot alter the GGUF route.
        CudaDriverApi.cuCtxGetDevice(out int device).ThrowOnError();
        CudaDriverApi.cuDeviceGetAttribute(out int major, 75, device).ThrowOnError();
        CudaDriverApi.cuDeviceGetAttribute(out int minor, 76, device).ThrowOnError();
        bool qualified = NinferQualifiedDefaultGeometry(true, _m.NLayer, _m.NEmbd, _m.NFf,
            _m.NVocab, _m.NHead, _m.NKvHead, _m.HeadDim, major, minor);
        _ninferQualifiedDefaults = PlanNinferQualifiedDefaults(qualified, _ninferAutoA16Option,
            _ninferAutoRepackOption, _ninferDecodeGraphsSetting.Value);
        // Prefill kernels are independent of verification/speculation and of the device's
        // SM count; they only need the Q24/KV4/HD256 recipe on CC 12.0 and their images.
        _ninferAutomaticPrefillAttention = qualified && _m.NUbatch >= 64
            && _m.NUbatch <= CudaNinferPrefillAttentionModule.MaximumRows
            && _m.NCtx >= _m.NUbatch && _m.NCtx <= CudaNinferPrefillAttentionModule.MaximumContext
            && Environment.GetEnvironmentVariable(CudaNinferPrefillAttentionModule.EnvironmentVariable) == null
            && Environment.GetEnvironmentVariable(CudaNinferAttention.PrefillThreadsEnvironment) == null
            && NinferAutomaticOptimizationRequested(Environment.GetEnvironmentVariable(NinferAutoPrefillAttentionEnvironment), NinferAutoPrefillAttentionEnvironment);
        _ninferAutomaticGdnChunk = qualified && _m.NUbatch >= 64
            && Environment.GetEnvironmentVariable(NinferGdnChunkEnvironment) == null
            && NinferAutomaticOptimizationRequested(Environment.GetEnvironmentVariable(NinferAutoGdnChunkEnvironment), NinferAutoGdnChunkEnvironment);
        _ninferDecodeGraphsEnabled = _ninferQualifiedDefaults.DecodeGraphs;
        NinferDecodeGraphsSelection = _ninferQualifiedDefaults.GraphSelection;
        NinferA16Selection = _ninferAutoA16Option == "0" ? "disabled" : qualified ? "auto-pending" : "unsupported-hardware";
        NinferLoadRepackSelection = _ninferAutoRepackOption == "0" ? "disabled" : qualified ? "auto-pending" : "unsupported-hardware";
        Console.Error.WriteLine($"[q35-cuda] NInfer qualified defaults: device=sm_{major}{minor}, graphs={NinferDecodeGraphsSelection}; automatic vector/repack require their packaged assets");
    }

    /// <summary>Use the normal PTX resolver. CudaModule.LoadFromFile gives a
    /// matching sibling fatbin preference and records the image actually loaded.</summary>
    internal static string LocateNinferAutomaticAsset(string filename, Func<string, string> locate)
        => locate(filename) ?? locate(Path.ChangeExtension(filename, ".fatbin"));

    internal static bool UseAutomaticNinferA16(bool automaticEnabled, string original, string legacyExact)
    {
        if (original != null && legacyExact != null)
            throw new ArgumentException("Select only one NInfer A16 arithmetic policy.");
        // Even a malformed explicit choice must be rejected by its existing
        // parser, rather than hidden by an automatically discovered image.
        return automaticEnabled && original == null && legacyExact == null;
    }

    // Only module admission may fall back. In particular, OOM, invalid device
    // context, device faults and every execution/launch failure remain errors.
    internal static bool NinferAutomaticModuleUnavailable(Exception error)
        => error is IOException or UnauthorizedAccessException or BadImageFormatException or InvalidOperationException
            || error is CudaException { ErrorCode: 200 or 209 or 218 or 222 or 500 };

    private void ReportNinferAutomaticFallback(string feature, Exception error)
        => Console.Error.WriteLine($"[q35-cuda] automatic NInfer {feature} unavailable before use: {error.Message}; using the existing implementation");

    private void ReportNinferA16Selection(string policy)
        => Console.Error.WriteLine($"[q35-cuda] NInfer A16 selection={NinferA16Selection}, policy={policy}, module={NinferA16ImageSha256}, path={NinferA16ModulePath}");
}
