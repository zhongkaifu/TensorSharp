// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;

namespace TensorSharp.Cuda;

public sealed unsafe partial class Q35CudaEngine
{
    internal const string NinferVerify8GemmEnvironment = "TS_Q35_NINFER_VERIFY8_GEMM";
    public string NinferVerify8GemmSelection { get; private set; } = "not-ninfer";
    public bool NinferVerify8GemmEnabled => _k?.NinferVerify8GemmEnabled == true;
    public string NinferVerify8GemmImageSha256 { get; private set; }
    public string NinferVerify8GemmPolicy => NinferVerify8GemmImageSha256 == null ? null : Q35Kernels.NinferVerify8GemmPolicy;
    /// <summary>Host submissions including capture, not actual GPU graph replays.</summary>
    public long NinferVerify8GemmHostSubmissions => _k?.NinferVerify8GemmHostSubmissions ?? 0;
    private string _ninferVerify8GemmSuffix = string.Empty;
    private string NinferVerify8GemmNamespaceSuffix => _ninferVerify8GemmSuffix;

    internal static bool PlanNinferVerify8Gemm(bool ninfer, bool qualified, int sms,
        bool verification, int maximumRows, string option)
    {
        if (!ninfer) return false; // The new option does not apply to GGUF, including invalid strings.
        bool requested = NinferAutomaticOptimizationRequested(option, NinferVerify8GemmEnvironment);
        bool supported = qualified && sms == 170 && verification && maximumRows >= 8;
        if (option == "1" && !supported)
            throw new NotSupportedException(NinferVerify8GemmEnvironment + "=1 requires the qualified Qwen3.8/CC12.0/170-SM target and enabled T8 verification.");
        return requested && supported;
    }

    private void InitializeNinferVerify8Gemm()
    {
        if (_m.Ninfer == null) return;
        string option = Environment.GetEnvironmentVariable(NinferVerify8GemmEnvironment);
        bool requested = PlanNinferVerify8Gemm(true, _ninferQualifiedDefaults?.Qualified == true,
            _sms, _ninferVerificationEnabled, _m.NUbatch, option);
        NinferVerify8GemmSelection = option == "0" ? "disabled" : requested ? "asset-missing" : "unsupported-profile";
        if (requested && _k.ConfigureNinferVerify8Gemm(option == "1"))
        {
            NinferVerify8GemmSelection = option == "1" ? "explicit" : "auto-sm120";
            NinferVerify8GemmImageSha256 = _k.ImageSha256;
            _ninferVerify8GemmSuffix = "|ninfer-verify8-gemm=" + Q35Kernels.NinferVerify8GemmPolicy
                + ":" + _k.ImageSha256.ToLowerInvariant();
        }
        Console.Error.WriteLine($"[q35-cuda] NInfer Verify8 GEMM selection={NinferVerify8GemmSelection}, enabled={NinferVerify8GemmEnabled}, policy={Q35Kernels.NinferVerify8GemmPolicy}, module={_k.ImageSha256}, rows=8,threads=256,smem=43520");
    }
}
