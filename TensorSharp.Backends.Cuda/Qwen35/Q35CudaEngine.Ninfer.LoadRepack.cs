// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.IO;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

public sealed unsafe partial class Q35CudaEngine
{
    internal const string NinferLoadRepackEnvironment = "TS_Q35_NINFER_REPACK_PTX";
    private CudaNinferRepackModule _ninferLoadRepack;
    public string NinferLoadRepackImageSha256 { get; private set; }
    public int NinferLoadRepackedMatrices { get; private set; }
    public long NinferLoadRepackedOriginalBytes { get; private set; }

    internal static string ResolveNinferLoadRepackPath(string value, bool ninfer)
    {
        if (value == null) return null;
        if (!ninfer)
            throw new NotSupportedException(NinferLoadRepackEnvironment + " requires an original NInfer target.");
        if (string.IsNullOrWhiteSpace(value) || !Path.IsPathFullyQualified(value))
            throw new ArgumentException(NinferLoadRepackEnvironment + " requires an absolute module path.");
        return Path.GetFullPath(value);
    }

    private void InitializeNinferLoadRepack()
    {
        string path = ResolveNinferLoadRepackPath(Environment.GetEnvironmentVariable(NinferLoadRepackEnvironment), _m.Ninfer != null);
        bool automatic = path == null && _ninferQualifiedDefaults?.AutoRepack == true;
        if (automatic)
        {
            path = LocateNinferAutomaticAsset(NinferQualifiedRepackAsset, CudaKernels.LocatePtxPath);
            if (path == null)
            {
                NinferLoadRepackSelection = "asset-missing";
                Console.Error.WriteLine("[q35-cuda] automatic lossless NInfer repack asset missing; using bounded CPU layout conversion");
            }
        }
        if (path == null) return;
        try { _ninferLoadRepack = new CudaNinferRepackModule(path); }
        catch (Exception error) when (automatic && NinferAutomaticModuleUnavailable(error))
        {
            NinferLoadRepackSelection = "asset-unavailable";
            ReportNinferAutomaticFallback("lossless GPU layout conversion", error);
            return;
        }
        NinferLoadRepackImageSha256 = _ninferLoadRepack.ImageSha256;
        NinferLoadRepackSelection = automatic ? "auto-sm120" : "explicit";
        NinferLoadRepackModulePath = path;
    }

    private void DisposeNinferLoadRepack()
    {
        var module = _ninferLoadRepack;
        _ninferLoadRepack = null;
        module?.Dispose();
    }

    private void CompleteNinferLoadRepack()
    {
        if (_ninferLoadRepack == null) return;
        Console.Error.WriteLine($"[q35-cuda] lossless NInfer GPU layout conversion: {NinferLoadRepackedMatrices} matrices, {NinferLoadRepackedOriginalBytes} original bytes, module={NinferLoadRepackImageSha256}");
        DisposeNinferLoadRepack(); // Only loading needs this module and the loader's staging.
    }

    private sealed partial class Loader
    {
        /// <summary>Moves original encoded bytes through the existing bounded staging
        /// buffer, then changes their placement on the GPU. The CPU source validates
        /// every original scale and calibration trailer before a kernel can consume it.
        /// Larger objects retain the existing bounded CPU conversion path.</summary>
        public bool TryRepackNinfer(CudaNinferProjectionDesc descriptor, IntPtr destination)
        {
            if (_e._ninferLoadRepack == null || descriptor.Format != CudaNinferProjectionFormat.NvFp4Planes
                || descriptor.Source is not ICudaNinferOriginalProjectionSource original
                || original.OriginalBytes > StagingBytes)
                return false;
            var source = new CudaNinferValidatedOriginalSource(descriptor);
            source.Read(0, _host, source.Bytes);
            source.RequireCompleteValidation();
            CudaDriverApi.cuMemcpyHtoD(_dev, _host, new UIntPtr(checked((ulong)source.Bytes))).ThrowOnError();
            _e._ninferLoadRepack.Repack(_dev, destination, descriptor.Rows, descriptor.Columns, original.RowMapping, _e.Stream);
            _e.Sync(); // Both staging buffers are reused by the next weight.
            _e.NinferLoadRepackedMatrices++;
            _e.NinferLoadRepackedOriginalBytes = checked(_e.NinferLoadRepackedOriginalBytes + source.Bytes);
            return true;
        }
    }
}
