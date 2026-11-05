// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;

namespace TensorSharp.Cuda;

public sealed unsafe partial class Q35CudaEngine
{
    internal const string NinferGateUpSwiGluEnvironment = "TS_Q35_NINFER_GATEUP_SWIGLU";
    public bool NinferGateUpSwiGluEnabled => _k?.NinferGateUpSwiGluEnabled == true;
    public string NinferGateUpSwiGluSelection { get; private set; } = "not-ninfer";
    public string NinferGateUpSwiGluImageSha256 { get; private set; }
    public long NinferGateUpSwiGluHostSubmissions => _k?.NinferGateUpSwiGluHostSubmissions ?? 0;
    public long NinferGateUpSwiGluCaptureCount { get; private set; }
    public long NinferGateUpSwiGluReplayCount { get; private set; }
    private string _ninferGateUpSwiGluSuffix = string.Empty;

    internal static bool PlanNinferGateUpSwiGlu(bool ninfer, bool qualified, int sms, int context,
        int maximumRows, bool verify, bool graphs, bool verify8Gemm, bool overrides, bool allGateUps, string option)
    {
        if (!ninfer) return false;
        bool requested = NinferAutomaticOptimizationRequested(option, NinferGateUpSwiGluEnvironment);
        // Context is not an operand of the fixed GateUp/SwiGLU specialization.
        bool supported = qualified && sms == 170 && (context is 40960 or 65536 or 131072 or 262144) && maximumRows >= 8
            && verify && graphs && verify8Gemm && !overrides && allGateUps;
        if (option == "1" && !supported)
            throw new NotSupportedException(NinferGateUpSwiGluEnvironment + "=1 requires qualified context40960 or context65536/170-SM T8 verification, graphs and Verify8 GEMM with original NVFP4 GateUp projections.");
        return requested && supported;
    }
    internal static bool NinferGateUpSwiGluMatrixEligible(CudaNinferProjectionFormat format, bool allowA4, int rows, int n, int k)
        => format == CudaNinferProjectionFormat.NvFp4Planes && allowA4 && rows == 8 && n == 34816 && k == 5120;

    private bool UsesNinferGateUpSwiGlu(RowGroup[] groups, int tapFirst)
        => groups.Length == 1 && UseNinferT8VerifyPass(NinferGateUpSwiGluEnabled, 1,
            groups[0].RowsPerSeq, groups[0].Slots.Length, groups[0].Verify, groups[0].Counts != null,
            _embedSpans != null, _featureCount != 0, _d != null, tapFirst);

    internal static void ValidateNinferGateUpSwiGluGraphIdentity(int lanes, int window, bool verify, int tapFirst, string identity)
    {
        if (identity == null) return;
        const string prefix = "|ninfer-gateup-swiglu=" + Q35Kernels.NinferGateUpSwiGluPolicy + ":";
        if (lanes != 1 || window != 8 || !verify || tapFirst != 0 || identity.Length != prefix.Length + 64
            || !identity.StartsWith(prefix, StringComparison.Ordinal)) throw new ArgumentException("Unqualified GateUp/SwiGLU graph identity.");
        for (int i = prefix.Length; i < identity.Length; i++)
            if (!(identity[i] is >= '0' and <= '9' or >= 'a' and <= 'f')) throw new ArgumentException("Non-normalized GateUp/SwiGLU image identity.");
    }

    private void InitializeNinferGateUpSwiGlu()
    {
        if (_m.Ninfer == null) return;
        // Per-matrix: FP8 GateUp layers take their own fused SwiGLU projection.
        bool all = false;
        foreach (var layer in _m.Ninfer.Layers)
            if (NinferGateUpSwiGluMatrixEligible(layer.GateUp.Format, layer.GateUp.AllowA4, 8,
                layer.GateUp.Rows, layer.GateUp.Columns)) all = true;
        string option = Environment.GetEnvironmentVariable(NinferGateUpSwiGluEnvironment);
        bool requested = PlanNinferGateUpSwiGlu(true, _ninferQualifiedDefaults?.Qualified == true, _sms,
            _m.NCtx, _m.NUbatch, _ninferVerificationEnabled, _ninferDecodeGraphsEnabled,
            NinferVerify8GemmEnabled, NinferFlashInferEnabled || NinferExactVerifyEnabled, all, option);
        NinferGateUpSwiGluSelection = option == "0" ? "disabled" : requested ? "asset-missing" : "unsupported-profile";
        if (requested && _k.ConfigureNinferGateUpSwiGlu(option == "1"))
        {
            NinferGateUpSwiGluSelection = option == "1" ? "explicit" : "auto-sm120-170sm";
            NinferGateUpSwiGluImageSha256 = _k.ImageSha256;
            _ninferGateUpSwiGluSuffix = "|ninfer-gateup-swiglu=" + Q35Kernels.NinferGateUpSwiGluPolicy
                + ":" + _k.ImageSha256.ToLowerInvariant();
        }
        Console.Error.WriteLine($"[q35-cuda] NInfer GateUp SwiGLU selection={NinferGateUpSwiGluSelection}, enabled={NinferGateUpSwiGluEnabled}, policy={Q35Kernels.NinferGateUpSwiGluPolicy}, module={NinferGateUpSwiGluImageSha256}, rows=8,lanes=1,context={_m.NCtx}");
    }

    private bool TryNinferGateUpSwiGlu(NinferMatrix matrix, int rows, bool pass)
    {
        var d = matrix.Descriptor;
        if (!pass || !NinferGateUpSwiGluEnabled
            || !NinferGateUpSwiGluMatrixEligible(d.Format, d.AllowA4, rows, d.Rows, d.Columns)) return false;
        QuantizeNinferInput(_xn, matrix, rows);
        float alpha = 1.0f / (d.InputScaleDivisor * d.WeightScaleDivisor);
        _k.GemmNinferGateUpSwiGlu(matrix.Data, matrix.Scales, _fp4c, _fp4s, _fp4r,
            alpha, _act, _partial, _counters, Stream);
        return true;
    }
}
