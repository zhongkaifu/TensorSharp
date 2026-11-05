// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.IO;

namespace TensorSharp.Cuda;

public sealed unsafe partial class Q35CudaEngine
{
    internal const string NinferA16ArithmeticPolicy = "original-k16-adjacent-t1c4-t23c1-f32-v1";
    internal const string NinferA16LegacyExactArithmeticPolicy = "legacy-k16-low8-high8-t1t3-c1-xor-f32-v1";
    private CudaNinferA16Module? _ninferA16Candidate;
    private CudaNinferA16LegacyExactModule? _ninferA16LegacyExactCandidate;
    private bool _ninferA16Initialized, _ninferA16Disposed;
    private string _ninferA16Suffix = string.Empty;
    private string NinferA16NamespaceSuffix => _ninferA16Suffix;

    /// <summary>Explicit paths retain their arithmetic policy. Otherwise the
    /// qualified SM120 default may load the packaged, legacy-exact vector image.</summary>
    private void InitializeNinferA16Candidate()
    {
        ObjectDisposedException.ThrowIf(_ninferA16Disposed, this);
        if (_ninferA16Initialized) return;
        string? raw = Environment.GetEnvironmentVariable(CudaNinferA16Module.EnvironmentVariable);
        string? exact = Environment.GetEnvironmentVariable(CudaNinferA16LegacyExactModule.EnvironmentVariable);
        bool automatic = UseAutomaticNinferA16(_ninferQualifiedDefaults?.AutoA16 == true, raw, exact);
        if (automatic)
        {
            exact = LocateNinferAutomaticAsset(NinferQualifiedA16Asset, CudaKernels.LocatePtxPath);
            if (exact == null)
            {
                NinferA16Selection = "asset-missing";
                Console.Error.WriteLine("[q35-cuda] automatic NInfer legacy-exact vector asset missing; using the existing A16 implementation");
            }
        }
        if (raw == null && exact == null) { _ninferA16Initialized = true; return; }
        if (_m.Ninfer == null)
            throw new NotSupportedException("NInfer A16 modules require the complete NInfer target body.");
        if (exact != null)
        {
            CudaNinferA16LegacyExactModule? module = null;
            try
            {
                try { module = CudaNinferA16LegacyExactModule.CreateExplicit(NormalizeNinferA16ModulePath(exact)!); }
                catch (Exception error) when (automatic && NinferAutomaticModuleUnavailable(error))
                {
                    NinferA16Selection = "asset-unavailable";
                    ReportNinferAutomaticFallback("legacy-exact vector", error);
                    _ninferA16Initialized = true;
                    return;
                }
                string suffix = BuildNinferA16StateSuffix(module.ImageSha256, NinferA16LegacyExactArithmeticPolicy);
                _ninferA16LegacyExactCandidate = module;
                _ninferA16Suffix = suffix;
                NinferA16Selection = automatic ? "auto-sm120" : "explicit-legacy-exact";
                NinferA16ModulePath = exact;
                NinferA16ImageSha256 = module.ImageSha256;
                _ninferA16Initialized = true;
                module = null;
                ReportNinferA16Selection(NinferA16LegacyExactArithmeticPolicy);
            }
            finally { module?.Dispose(); }
            return;
        }
        string path = NormalizeNinferA16ModulePath(raw)!;
        CudaNinferA16Module? candidate = null;
        try
        {
            candidate = CudaNinferA16Module.CreateExplicit(path)!;
            string suffix = BuildNinferA16StateSuffix(candidate.ImageSha256);
            _ninferA16Candidate = candidate;
            _ninferA16Suffix = suffix;
            NinferA16Selection = "explicit-original";
            NinferA16ModulePath = path;
            NinferA16ImageSha256 = candidate.ImageSha256;
            _ninferA16Initialized = true;
            candidate = null; // Ownership transfers only after path/image/policy validation.
            ReportNinferA16Selection(NinferA16ArithmeticPolicy);
        }
        finally { candidate?.Dispose(); }
    }

    internal static string? NormalizeNinferA16ModulePath(string? raw)
    {
        if (raw == null) return null;
        if (string.IsNullOrWhiteSpace(raw) || !Path.IsPathFullyQualified(raw) || raw != Path.GetFullPath(raw))
            throw new ArgumentException(CudaNinferA16Module.EnvironmentVariable + " requires an explicit normalized absolute PTX/fatbin path.", nameof(raw));
        return raw;
    }

    internal static string BuildNinferA16StateSuffix(string imageSha256, string arithmeticPolicy = NinferA16ArithmeticPolicy)
    {
        if (arithmeticPolicy is not (NinferA16ArithmeticPolicy or NinferA16LegacyExactArithmeticPolicy))
            throw new ArgumentException("Unknown NInfer A16 arithmetic policy.", nameof(arithmeticPolicy));
        if (imageSha256 == null || imageSha256.Length != 64)
            throw new ArgumentException("The NInfer A16 namespace requires a complete module SHA256.", nameof(imageSha256));
        foreach (char c in imageSha256)
            if (!Uri.IsHexDigit(c)) throw new ArgumentException("Invalid module SHA256.", nameof(imageSha256));
        return "|ninfer-a16=" + arithmeticPolicy + ":" + imageSha256.ToLowerInvariant();
    }

    /// <summary>Only replaces the caller's existing A16 branch, after its A4/BF16 selection.
    /// F32 output retains the original schedule accumulator for the caller's epilogue.</summary>
    private bool TryNinferA16(NinferMatrix matrix, IntPtr input, IntPtr output, int rows)
    {
        ObjectDisposedException.ThrowIf(_ninferA16Disposed, this);
        var candidate = _ninferA16Candidate;
        if ((candidate == null && _ninferA16LegacyExactCandidate == null) || rows < 1 || rows > 3) return false;
        var descriptor = matrix?.Descriptor;
        if (descriptor == null || descriptor.Format != CudaNinferProjectionFormat.NvFp4Planes
            || descriptor.Rows < 1 || descriptor.Columns is not (5120 or 6144 or 17408))
            return false;
        float inverseWeightDivisor = 1.0f / descriptor.WeightScaleDivisor;
        if (_ninferA16LegacyExactCandidate != null)
            _ninferA16LegacyExactCandidate.Project(matrix.Data, matrix.Scales, input, output, descriptor.Rows, descriptor.Columns,
                rows, inverseWeightDivisor, Stream, inputStride: descriptor.Columns, outputStride: descriptor.Rows);
        else
            candidate.Project(matrix.Data, matrix.Scales, input, output, descriptor.Rows, descriptor.Columns,
                rows, inverseWeightDivisor, Stream, inputStride: descriptor.Columns, outputStride: descriptor.Rows);
        return true;
    }

    private void DisposeNinferA16Candidate()
    {
        if (_ninferA16Disposed) return;
        _ninferA16Disposed = true;
        var candidate = _ninferA16Candidate;
        _ninferA16Candidate = null;
        candidate?.Dispose();
        _ninferA16LegacyExactCandidate?.Dispose();
        _ninferA16LegacyExactCandidate = null;
        // Keep the immutable namespace available for post-disposal diagnostics.
    }
}
