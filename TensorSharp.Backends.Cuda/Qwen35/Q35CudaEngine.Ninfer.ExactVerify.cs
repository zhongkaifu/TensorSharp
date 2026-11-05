// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#nullable enable
using System;
using System.IO;

namespace TensorSharp.Cuda;

public sealed unsafe partial class Q35CudaEngine
{
    internal const string NinferExactVerifyEnvironment = "TS_Q35_NINFER_EXACT_VERIFY_PTX";
    internal const int NinferExactVerifyRowLimit = 8;
    internal const string NinferExactVerifyArithmeticPolicy = "legacy-k16-low8-high8-c1-xor-verify-nv4-mina4gt1-t1t8-f32-v1";

    private CudaNinferA16BatchExactModule? _ninferExactVerify;
    private bool _ninferExactVerifyInitialized, _ninferExactVerifyDisposed;
    private string _ninferExactVerifySuffix = string.Empty;

    // The configured arithmetic identity remains available after disposal. The
    // module itself is never discovered or enabled by the ordinary A16 options.
    private string NinferExactVerifyNamespaceSuffix => _ninferExactVerifySuffix;
    public bool NinferExactVerifyEnabled => _ninferExactVerifySuffix.Length != 0;
    public int NinferExactVerifyMaximumRows => NinferExactVerifyEnabled ? NinferExactVerifyRowLimit : MaxVerifyRows;

    /// <summary>Snapshots the explicit verification-only image once. This does not
    /// qualify full batched target equivalence: BF16 projections, attention and the
    /// head retain their own existing batched arithmetic.</summary>
    private void InitializeNinferExactVerify()
    {
        ObjectDisposedException.ThrowIf(_ninferExactVerifyDisposed, this);
        if (_ninferExactVerifyInitialized) return;
        string? path = ResolveNinferExactVerifyPath(Environment.GetEnvironmentVariable(NinferExactVerifyEnvironment));
        if (path == null) { _ninferExactVerifyInitialized = true; return; }
        if (_m.Ninfer == null || !_ninferVerificationEnabled)
            throw new NotSupportedException("Exact NInfer verification requires the complete NInfer body and explicit "
                + NinferVerificationEnvironment + "=1 option.");

        CudaNinferA16BatchExactModule? module = null;
        try
        {
            module = new CudaNinferA16BatchExactModule(path);
            string suffix = BuildNinferExactVerifyStateSuffix(module.ImageSha256);
            _ninferExactVerify = module;
            _ninferExactVerifySuffix = suffix;
            _ninferExactVerifyInitialized = true;
            module = null;
        }
        finally { module?.Dispose(); }
    }

    internal static string? ResolveNinferExactVerifyPath(string? raw)
    {
        if (raw == null) return null;
        if (string.IsNullOrWhiteSpace(raw) || !Path.IsPathFullyQualified(raw) || raw != Path.GetFullPath(raw))
            throw new ArgumentException(NinferExactVerifyEnvironment + " requires an explicit normalized absolute PTX/fatbin path.", nameof(raw));
        if (!File.Exists(raw))
            throw new FileNotFoundException("The explicit NInfer exact-verification module does not exist.", raw);
        return raw;
    }

    internal static string BuildNinferExactVerifyStateSuffix(string imageSha256)
    {
        if (imageSha256 == null || imageSha256.Length != 64)
            throw new ArgumentException("The NInfer exact-verification namespace requires a complete module SHA256.", nameof(imageSha256));
        foreach (char c in imageSha256)
            if (!Uri.IsHexDigit(c)) throw new ArgumentException("Invalid module SHA256.", nameof(imageSha256));
        return "|ninfer-exact-verify=" + NinferExactVerifyArithmeticPolicy + ":" + imageSha256.ToLowerInvariant()
            + ":maxrows=" + NinferExactVerifyRowLimit;
    }

    /// <summary>Must be called by Verify before cache room allocation, checkpoint
    /// invalidation, slot binding, uploads, or any other state mutation.</summary>
    private void ValidateNinferExactVerifyRequest(int lanes, int window)
        => ValidateNinferExactVerifyGeometry(NinferExactVerifyEnabled, lanes, window);

    internal static void ValidateNinferExactVerifyGeometry(bool enabled, int lanes, int window)
    {
        if (!enabled) return;
        if (lanes < 1 || window < 1 || (long)lanes * window > NinferExactVerifyRowLimit)
            throw new ArgumentOutOfRangeException(nameof(window), "Exact NInfer verification supports at most eight total physical rows.");
    }

    internal static bool ShouldUseNinferExactVerify(bool enabled, bool isVerify,
        CudaNinferProjectionFormat format, int outputRows, int columns, int rows, int a4Threshold)
    {
        if (!enabled || !isVerify) return false;
        // Check the complete pass geometry before excluding a particular matrix.
        // In particular the first GDN input must not bypass the row-limit guard.
        ValidateNinferExactVerifyGeometry(true, 1, rows);
        if (format != CudaNinferProjectionFormat.NvFp4Planes || a4Threshold <= 1) return false;
        if (outputRows < 1 || columns is not (5120 or 6144 or 17408))
            throw new NotSupportedException("The enabled exact-verification policy has no legacy-order A16 kernel for this projection.");
        return true;
    }

    /// <summary>Invoke before the ordinary BF16/A4/A16 selection, with the actual
    /// pass flag and total physical rows. GDN input (a4Threshold=1) stays A4. The
    /// output is unrounded F32; the existing caller still owns its BF16 epilogue.</summary>
    private bool TryNinferExactVerify(NinferMatrix matrix, IntPtr input, IntPtr output,
        int rows, int a4Threshold, bool isVerify)
    {
        ObjectDisposedException.ThrowIf(_ninferExactVerifyDisposed, this);
        if (!NinferExactVerifyEnabled || !isVerify) return false;
        ArgumentNullException.ThrowIfNull(matrix);
        var d = matrix.Descriptor;
        if (!ShouldUseNinferExactVerify(true, true, d.Format, d.Rows, d.Columns, rows, a4Threshold)) return false;
        var module = _ninferExactVerify ?? throw new InvalidOperationException("The configured exact-verification module is unavailable.");
        module.Project(matrix.Data, matrix.Scales, input, output, d.Rows, d.Columns,
            rows, 1.0f / d.WeightScaleDivisor, Stream, inputStride: d.Columns, outputStride: d.Rows);
        return true;
    }

    private void DisposeNinferExactVerify()
    {
        if (_ninferExactVerifyDisposed) return;
        _ninferExactVerifyDisposed = true;
        var module = _ninferExactVerify;
        _ninferExactVerify = null;
        module?.Dispose();
    }
}
