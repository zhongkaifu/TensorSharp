// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;

namespace TensorSharp.Cuda;

public sealed unsafe partial class Q35CudaEngine
{
    internal const string NinferResidualNormEnvironment = "TS_Q35_NINFER_RESIDUAL_NORM";
    internal enum NinferLayerPass { Legacy, FusedResidualNorm }
    internal enum NinferResidualState { InitialNoDelta, LegacyDelta, FusedCanonical }
    internal enum NinferNormRoute { LegacyInitial, LegacyDelta, ReadOnlyCanonical }
    public bool NinferResidualNormEnabled { get; private set; }
    public string NinferResidualNormSelection { get; private set; } = "not-ninfer";
    public string NinferResidualNormImageSha256 { get; private set; }
    public string NinferResidualNormPolicy => NinferResidualNormEnabled ? Q35Kernels.NinferResidualNormPolicy : null;
    /// <summary>Host submissions, including capture; not GPU graph replay/completion counts.</summary>
    public long NinferResidualNormGemmHostSubmissions => _k?.NinferResidualGemmHostSubmissions ?? 0;
    public long NinferResidualNormPlainHostSubmissions => _k?.NinferPlainNormHostSubmissions ?? 0;
    public long NinferResidualNormInitialHostSubmissions { get; private set; }
    public long NinferResidualNormLegacyPostHostSubmissions { get; private set; }
    public long NinferResidualNormCaptureCount { get; private set; }
    public long NinferResidualNormReplayCount { get; private set; }
    public long NinferResidualNormCompletedVerifies { get; private set; }
    private string _ninferResidualNormSuffix = string.Empty;
    private string NinferResidualNormNamespaceSuffix => _ninferResidualNormSuffix;

    internal static bool PlanNinferResidualNorm(bool ninfer, bool qualified, int sms, int context,
        int maximumRows, bool verification, bool graphs, bool gemm8, bool splitK,
        bool flashInfer, bool exactVerify, string option, bool allDowns = true)
    {
        if (!ninfer) return false;
        bool requested = NinferAutomaticOptimizationRequested(option, NinferResidualNormEnvironment);
        // Configured context does not enter this fixed T8 projection/norm math.
        // Attention independently keeps its actual-visible 40960-key cap and fallback.
        bool supported = qualified && sms == 170 && (context is 40960 or 65536 or 131072 or 262144) && maximumRows >= 8
            && verification && graphs && gemm8 && splitK && !flashInfer && !exactVerify && allDowns;
        if (option == "1" && !supported)
            throw new NotSupportedException(NinferResidualNormEnvironment
                + "=1 requires the qualified context40960 or context65536/170-SM target, Verify8 GEMM/split-K and decode graphs, and at least one NVFP4 Down projection, without projection overrides.");
        return requested && supported;
    }

    internal static NinferLayerPass SelectNinferResidualPass(bool enabled, int groups, int window,
        int lanes, bool verify, bool counts, bool embeddings, bool featureTaps, bool ggufDrafter, int tapFirst = 0)
        => enabled && groups == 1 && window == 8 && lanes == 1 && verify && !counts && !embeddings
            && !featureTaps && !ggufDrafter && tapFirst == 0 ? NinferLayerPass.FusedResidualNorm : NinferLayerPass.Legacy;

    internal static bool NinferResidualMatrixEligible(CudaNinferProjectionFormat format, bool allowA4,
        int rows, int outputRows, int columns)
        => format == CudaNinferProjectionFormat.NvFp4Planes && allowA4
            && Q35Kernels.NinferResidualProjectionShape(rows, outputRows, columns);

    internal static NinferResidualState NinferProjectedResidualState(bool actuallyFused)
        => actuallyFused ? NinferResidualState.FusedCanonical : NinferResidualState.LegacyDelta;

    internal static NinferNormRoute PlanNinferResidualNormalization(NinferResidualState state,
        bool candidatePass, bool postProjection) => state switch
    {
        NinferResidualState.InitialNoDelta when !postProjection => NinferNormRoute.LegacyInitial,
        NinferResidualState.LegacyDelta => NinferNormRoute.LegacyDelta,
        NinferResidualState.FusedCanonical when candidatePass => NinferNormRoute.ReadOnlyCanonical,
        NinferResidualState.InitialNoDelta or NinferResidualState.FusedCanonical
            => throw new InvalidOperationException("The residual state does not permit this normalization transition."),
        _ => throw new ArgumentOutOfRangeException(nameof(state))
    };

    internal static string BuildNinferResidualNormStateSuffix(string imageSha)
    {
        if (imageSha?.Length != 64) throw new ArgumentException("An actual residual/norm module SHA256 is required.");
        foreach (char c in imageSha) if (!Uri.IsHexDigit(c)) throw new ArgumentException("Malformed residual/norm module SHA256.");
        return "|ninfer-residual-norm=" + Q35Kernels.NinferResidualNormPolicy + ":" + imageSha.ToLowerInvariant();
    }

    private void InitializeNinferResidualNorm()
    {
        if (_m.Ninfer == null) return;
        string option = Environment.GetEnvironmentVariable(NinferResidualNormEnvironment);
        // This immutable constructor-time check guarantees the final Down can
        // complete the fused pass. An automatic selector never changes policy
        // after work/graph capture, including when an older image lacks entries.
        // Per-matrix: an FP8 Down keeps the legacy delta and the next norm adds it.
        bool allDowns = false;
        foreach (var layer in _m.Ninfer.Layers)
            if (NinferResidualMatrixEligible(layer.Down.Format, layer.Down.AllowA4, 8, layer.Down.Rows, layer.Down.Columns))
                allDowns = true;
        bool requested = PlanNinferResidualNorm(true, _ninferQualifiedDefaults?.Qualified == true,
            _sms, _m.NCtx, _m.NUbatch, _ninferVerificationEnabled, _ninferDecodeGraphsEnabled,
            NinferVerify8GemmEnabled, Q35Kernels.NinferResidualNormSplitKEnabled,
            NinferFlashInferEnabled, NinferExactVerifyEnabled, option, allDowns);
        NinferResidualNormSelection = option == "0" ? "disabled" : requested ? "asset-missing" : "unsupported-profile";
        if (requested && _k.ConfigureNinferResidualNorm(option == "1"))
        {
            NinferResidualNormSelection = option == "1" ? "explicit" : "auto-sm120-170sm";
            NinferResidualNormImageSha256 = _k.ImageSha256;
            _ninferResidualNormSuffix = BuildNinferResidualNormStateSuffix(_k.ImageSha256);
            NinferResidualNormEnabled = true;
        }
        Console.Error.WriteLine($"[q35-cuda] NInfer residual norm selection={NinferResidualNormSelection}, enabled={NinferResidualNormEnabled}, policy={Q35Kernels.NinferResidualNormPolicy}, module={NinferResidualNormImageSha256}, rows=8,lanes=1,context={_m.NCtx}");
    }

    private bool TryNinferFusedResidualProjection(NinferMatrix matrix, IntPtr input, IntPtr output,
        int rows, bool candidatePass)
    {
        var d = matrix.Descriptor;
        if (!candidatePass || !NinferResidualNormEnabled
            || !NinferResidualMatrixEligible(d.Format, d.AllowA4, rows, d.Rows, d.Columns)) return false;
        // This directly selects the admitted A4 implementation. FI/ExactVerify
        // were refused at construction; no requested flag can stand in for a launch.
        QuantizeNinferInput(input, matrix, rows);
        float alpha = 1.0f / (d.InputScaleDivisor * d.WeightScaleDivisor);
        _k.GemmNinferResidual(matrix.Data, matrix.Scales, d.Rows, d.Columns, _fp4c, _fp4s, _fp4r,
            rows, alpha, output, _partial, _counters, _x, Stream);
        return true;
    }

    private NinferResidualState ProjectNinferResidual(NinferMatrix matrix, IntPtr input, int rows,
        bool isVerify, bool candidatePass)
    {
        bool fused = TryNinferFusedResidualProjection(matrix, input, _block, rows, candidatePass);
        if (!fused) NinferProject(matrix, input, _block, rows, round: false, isVerify: isVerify);
        return NinferProjectedResidualState(fused);
    }

    private void NormalizeNinferResidual(NinferResidualState state, IntPtr weight, int rows,
        bool candidatePass, bool postProjection, NinferMatrix quantizeFor = null)
    {
        if (TryNormalizeNinferPrefill(state, weight, rows, candidatePass, quantizeFor))
            return;
        switch (PlanNinferResidualNormalization(state, candidatePass, postProjection))
        {
            case NinferNormRoute.LegacyInitial:
                _ninfer.AddRmsNorm(_x, IntPtr.Zero, weight, _xn, rows, _m.NEmbd, _m.Eps, Stream);
                if (candidatePass) NinferResidualNormInitialHostSubmissions++;
                break;
            case NinferNormRoute.LegacyDelta:
                _ninfer.AddRmsNorm(_x, _block, weight, _xn, rows, _m.NEmbd, _m.Eps, Stream);
                if (candidatePass && postProjection) NinferResidualNormLegacyPostHostSubmissions++;
                break;
            case NinferNormRoute.ReadOnlyCanonical:
                _k.PlainNinferNorm(_x, weight, _xn, rows, _m.NEmbd, _m.Eps, Stream);
                break;
            default: throw new ArgumentOutOfRangeException(nameof(state));
        }
    }

    private void LogNinferResidualNormSummary()
    {
        if (NinferResidualNormEnabled)
            Console.Error.WriteLine($"[q35-cuda] NInfer residual norm host submissions: gemm={NinferResidualNormGemmHostSubmissions}, plain={NinferResidualNormPlainHostSubmissions}, initial={NinferResidualNormInitialHostSubmissions}, legacyPost={NinferResidualNormLegacyPostHostSubmissions}; graph captures={NinferResidualNormCaptureCount}, graph launches={NinferResidualNormReplayCount}, completedVerify={NinferResidualNormCompletedVerifies} (capture included; host counts are not GPU kernel replay/completion counts), module={NinferResidualNormImageSha256}");
    }
}
