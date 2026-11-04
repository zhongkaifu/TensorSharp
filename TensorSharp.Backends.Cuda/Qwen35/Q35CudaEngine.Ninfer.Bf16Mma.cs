// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.IO;

namespace TensorSharp.Cuda;

public sealed unsafe partial class Q35CudaEngine
{
    internal const string NinferBf16MmaEnvironment = "TS_Q35_NINFER_BF16_MMA";
    internal const string NinferBf16MmaAsset = "tensorsharp_ninfer_bf16_mma.ptx";
    private CudaNinferBf16MmaModule _ninferBf16Mma;
    private IntPtr _ninferBf16MmaInput;
    private string _ninferBf16MmaSuffix = string.Empty;
    public bool NinferBf16MmaEnabled => _ninferBf16Mma != null;
    public string NinferBf16MmaSelection { get; private set; } = "not-ninfer";
    public string NinferBf16MmaImageSha256 { get; private set; }
    public string NinferBf16MmaPolicy => NinferBf16MmaImageSha256 == null ? null : CudaNinferBf16MmaModule.ArithmeticPolicy;
    /// <summary>Host submissions including capture, not individual GPU graph replay counts.</summary>
    public long NinferBf16MmaHostSubmissions { get; private set; }
    public long NinferBf16MmaCaptureCount { get; private set; }
    public long NinferBf16MmaReplayCount { get; private set; }
    private string NinferBf16MmaNamespaceSuffix => _ninferBf16MmaSuffix;

    internal static bool PlanNinferBf16Mma(bool ninfer, bool qualified, int sms, int context,
        int maximumRows, bool verify, bool graphs, bool flashInfer, bool exact, string option)
    {
        if (!ninfer) return false;
        bool requested = NinferAutomaticOptimizationRequested(option, NinferBf16MmaEnvironment);
        // Only the configured profile widens; the two T8 shapes and scratch stay fixed.
        bool supported = qualified && sms == 170 && (context is 40960 or 65536) && maximumRows >= 8
            && verify && graphs && !flashInfer && !exact;
        if (option == "1" && !supported)
            throw new NotSupportedException(NinferBf16MmaEnvironment + "=1 requires the qualified 170-SM/context40960 or context65536 NInfer recipe, enabled verification/graphs, without FI or ExactVerify.");
        return requested && supported;
    }
    internal static bool UseNinferBf16MmaPass(bool enabled, int groups, int window, int lanes,
        bool verify, bool counts, bool embeddings, bool featureTaps, bool ggufDrafter, int tapFirst)
        => enabled && groups == 1 && window == 8 && lanes == 1 && verify && !counts
            && !embeddings && !featureTaps && !ggufDrafter && tapFirst == 0;
    internal static bool NinferBf16MmaProjectionEligible(bool pass, CudaNinferProjectionFormat format,
        int rows, int n, int k) => pass && format == CudaNinferProjectionFormat.BFloat16 && rows == 8
            && ((n == 14336 && k == 5120) || (n == 5120 && k == 6144));
    internal static string BuildNinferBf16MmaStateSuffix(string sha)
    {
        if (sha?.Length != 64) throw new ArgumentException("The actual BF16 MMA module identity is required.");
        foreach (char c in sha) if (!Uri.IsHexDigit(c)) throw new ArgumentException("Malformed BF16 MMA module SHA256.");
        return "|ninfer-bf16-mma=" + CudaNinferBf16MmaModule.ArithmeticPolicy + ":" + sha.ToLowerInvariant();
    }
    internal static void ValidateNinferBf16MmaGraphIdentity(int lanes, int window, bool verify, int tapFirst, string identity)
    {
        if (identity == null) return;
        const string prefix = "|ninfer-bf16-mma=" + CudaNinferBf16MmaModule.ArithmeticPolicy + ":";
        if (lanes != 1 || window != 8 || !verify || tapFirst != 0 || identity.Length != prefix.Length + 64
            || !identity.StartsWith(prefix, StringComparison.Ordinal)) throw new ArgumentException("Unqualified BF16 MMA graph identity.");
        for (int i = prefix.Length; i < identity.Length; i++)
            if (!(identity[i] is >= '0' and <= '9' or >= 'a' and <= 'f')) throw new ArgumentException("Non-normalized BF16 MMA graph image identity.");
    }
    private bool UsesNinferBf16Mma(RowGroup[] groups, int tapFirst)
        => groups.Length == 1 && UseNinferBf16MmaPass(NinferBf16MmaEnabled, 1,
            groups[0].RowsPerSeq, groups[0].Slots.Length, groups[0].Verify, groups[0].Counts != null,
            _embedSpans != null, _featureCount != 0, _d != null, tapFirst);

    internal static string ResolveNinferBf16MmaAsset(bool required, Func<string, string> locate)
    {
        string path = LocateNinferAutomaticAsset(NinferBf16MmaAsset, locate);
        if (path == null && required)
            throw new FileNotFoundException(NinferBf16MmaEnvironment + "=1 requires the separate " + NinferBf16MmaAsset + " CUDA asset.");
        return path;
    }
    // Only missing filesystem assets may fall back before publication. A present
    // but incompatible/invalid module, CUDA context fault or allocation failure
    // must stay visible; no execution-time fallback is permitted.
    internal static bool NinferBf16MmaAssetMissing(Exception error)
        => error is FileNotFoundException or DirectoryNotFoundException;

    private void InitializeNinferBf16Mma()
    {
        if (_m.Ninfer == null) return;
        string option = Environment.GetEnvironmentVariable(NinferBf16MmaEnvironment);
        bool requested = PlanNinferBf16Mma(true, _ninferQualifiedDefaults?.Qualified == true, _sms,
            _m.NCtx, _m.NUbatch, _ninferVerificationEnabled, _ninferDecodeGraphsEnabled,
            NinferFlashInferEnabled, NinferExactVerifyEnabled, option);
        NinferBf16MmaSelection = option == "0" ? "disabled" : requested ? "asset-missing" : "unsupported-profile";
        if (!requested) { LogNinferBf16MmaSelection(); return; }
        string path = ResolveNinferBf16MmaAsset(option == "1", CudaKernels.LocatePtxPath);
        if (path == null) { LogNinferBf16MmaSelection(); return; }
        CudaNinferBf16MmaModule module = null;
        try
        {
            try { module = new CudaNinferBf16MmaModule(path); }
            catch (Exception error) when (option == null && NinferBf16MmaAssetMissing(error))
            {
                LogNinferBf16MmaSelection();
                return; // Asset disappeared before use; no module/graph state was published.
            }
            IntPtr scratch = Alloc(CudaNinferBf16MmaModule.ScratchBytes);
            string suffix = BuildNinferBf16MmaStateSuffix(module.ImageSha256);
            _ninferBf16Mma = module; _ninferBf16MmaInput = scratch;
            NinferBf16MmaImageSha256 = module.ImageSha256; _ninferBf16MmaSuffix = suffix;
            NinferBf16MmaSelection = option == "1" ? "explicit" : "auto-sm120-170sm";
            module = null;
        }
        finally { module?.Dispose(); }
        LogNinferBf16MmaSelection();
    }
    private void LogNinferBf16MmaSelection()
        => Console.Error.WriteLine($"[q35-cuda] NInfer BF16 MMA selection={NinferBf16MmaSelection}, enabled={NinferBf16MmaEnabled}, policy={CudaNinferBf16MmaModule.ArithmeticPolicy}, module={NinferBf16MmaImageSha256}, rows=8,lanes=1,context={_m.NCtx},scratch={(NinferBf16MmaEnabled ? CudaNinferBf16MmaModule.ScratchBytes : 0)}");
    private bool TryNinferBf16Mma(NinferMatrix matrix, IntPtr input, IntPtr output, int rows, bool pass)
    {
        var d = matrix.Descriptor;
        if (!NinferBf16MmaProjectionEligible(pass, d.Format, rows, d.Rows, d.Columns)) return false;
        var module = _ninferBf16Mma ?? throw new InvalidOperationException("BF16 MMA policy has no live owning module.");
        var plan = CudaNinferBf16MmaModule.Plan(d.Rows, d.Columns, rows);
        CudaNinferBf16MmaModule.ValidateBuffers(_ninferBf16MmaInput, matrix.Data, output, plan);
        _ninfer.CastToBf16(input, _ninferBf16MmaInput, (long)rows * d.Columns, Stream);
        module.Project(_ninferBf16MmaInput, matrix.Data, output, d.Rows, d.Columns, rows, Stream);
        NinferBf16MmaHostSubmissions++;
        return true; // The caller retains its original Round or F32 residual consumer.
    }
    private void DisposeNinferBf16Mma()
    {
        if (_ninferBf16Mma != null)
            Console.Error.WriteLine($"[q35-cuda] NInfer BF16 MMA host submissions={NinferBf16MmaHostSubmissions}, graph captures={NinferBf16MmaCaptureCount}, graph launches={NinferBf16MmaReplayCount} (capture included; not individual GPU kernel replay counts), module={NinferBf16MmaImageSha256}");
        var module = _ninferBf16Mma; _ninferBf16Mma = null; module?.Dispose();
    }
}
