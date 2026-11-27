// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Threading;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

/// <summary>Optional model-owned LTX K-quant projections. Small-row products
/// retain Float32 arithmetic; the optional prefill converter preserves the
/// existing Float16 operand policy. Older PTX keeps the existing CUDA route.</summary>
public sealed class CudaLtx25ProjectionSession : IDisposable
{
    private readonly CudaAllocator _allocator;
    private readonly object _sync = new();
    private CudaModule _module;
    private IntPtr _tile8, _tile16, _dequantF16;
    private Tensor _weightF16, _activationF16;
    private bool _disposed;
    private long _executed;
    private long _f16Executed, _f16Converted;
    public string ModulePath { get; }
    public bool IsAvailable => _module != null;
    public long ExecutedProjections => Interlocked.Read(ref _executed);
    public bool F16DequantAvailable => _module != null && _dequantF16 != IntPtr.Zero;
    public long ExecutedF16Projections => Interlocked.Read(ref _f16Executed);
    public long ExecutedF16Dequantizations => Interlocked.Read(ref _f16Converted);
    public long F16ScratchBytes => ((_weightF16?.ElementCount() ?? 0) + (_activationF16?.ElementCount() ?? 0)) * sizeof(ushort);

    public CudaLtx25ProjectionSession(CudaAllocator allocator, bool requireNative = false, bool enableF16 = false)
    {
        _allocator = allocator ?? throw new ArgumentNullException(nameof(allocator));
        ModulePath = CudaKernels.LocatePtxPath("tensorsharp_ltx25.ptx");
        try
        {
            if (ModulePath == null) throw new NotSupportedException("tensorsharp_ltx25.ptx was not found.");
            allocator.Context.MakeCurrent();
            _module = CudaModule.LoadFromFile(ModulePath);
            _tile8 = _module.GetFunction("ts_ltx25_quant_tile8_f32");
            _tile16 = _module.GetFunction("ts_ltx25_quant_tile16_f32");
            try { _dequantF16 = _module.GetFunction("ts_ltx25_dequant_k_f16"); }
            // A tile-only older module still serves the small-row products; a caller that requires the F16 route fails.
            catch when (!(requireNative && enableF16)) { _dequantF16 = IntPtr.Zero; }
        }
        catch (Exception error)
        {
            _module?.Dispose(); _module = null;
            if (requireNative) throw new NotSupportedException("Required LTX CUDA projection module could not be loaded: " + ModulePath, error);
        }
    }

    /// <summary>Decode unchanged K-quant bytes to [columns,inner] Float16 using
    /// the existing Float32 decode and nearest-even Half rounding. Older PTX or
    /// a disabled optional F16 route returns false before dispatch.</summary>
    public unsafe bool TryDequantizeF16(Tensor output, IntPtr host, int type, long inner, long columns, long bytes)
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            int blockBytes = type switch { 12 => 144, 13 => 176, 14 => 210, _ => 0 };
            if (!F16DequantAvailable || output == null || host == IntPtr.Zero || blockBytes == 0 ||
                inner <= 0 || inner > int.MaxValue || inner % 256 != 0 || columns <= 0 || columns > int.MaxValue ||
                bytes != checked(inner / 256 * columns * blockBytes) || output.DimensionCount != 2 ||
                output.Sizes[0] != columns || output.Sizes[1] != inner ||
                !CudaKernelOps.TryGetContiguousFloatOrHalf(output, out var os, out var yp, out _, out bool half) ||
                !half || !ReferenceEquals(os.AllocatorImpl, _allocator)) return false;
            _allocator.Context.MakeCurrent();
            CudaQuantizedOps.PreloadQuantizedWeight(_allocator, host, host, type, inner, columns, bytes);
            if (!CudaQuantizedOps.TryGetResidentDevicePtr(_allocator, host, out var weight, out int storedType) || storedType != type) return false;
            if (output.StorageOffset != 0 || output.ElementCount() != output.Storage.ElementCount) os.EnsureDeviceCurrent();
            long blocks = inner / 256 * columns;
            void** args = stackalloc void*[4] { &weight, &yp, &type, &blocks };
            CudaDriverApi.cuLaunchKernel(_dequantF16, checked((uint)((blocks + 7) / 8)), 1, 1,
                256, 1, 1, 0, _allocator.Stream.Handle, (IntPtr)args, IntPtr.Zero).ThrowOnError();
            os.MarkDeviceModified();
            Interlocked.Increment(ref _f16Converted);
            return true;
        }
    }

    /// <summary>Optional blockwise converter feeding the same F16 cuBLAS policy
    /// as the existing eligible prefill route. Scratch belongs to this model
    /// session; decoded weights are not retained in a shared mutable cache.</summary>
    public bool TryPrefillLinear(Tensor output, Tensor input, IntPtr host, int type, long inner, long columns, long bytes)
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (!F16DequantAvailable || host == IntPtr.Zero || type is not (12 or 13 or 14) ||
                inner <= 0 || inner > int.MaxValue || inner % 256 != 0 || columns <= 0 || columns > int.MaxValue ||
                bytes != checked(inner / 256 * columns * (type == 12 ? 144 : type == 13 ? 176 : 210)) ||
                input == null || output == null || input.DimensionCount != 2 || output.DimensionCount != 2 ||
                input.Sizes[1] != inner || output.Sizes[1] != columns || input.Sizes[0] != output.Sizes[0] ||
                input.Sizes[0] < CudaQuantizedOps.F16GemmMinRows || input.Sizes[0] > int.MaxValue ||
                2L * inner * columns > CudaQuantizedOps.F16GemmMaxWeightBytes ||
                !CudaKernelOps.TryGetContiguousFloat(input, out var xs, out var xp, out int inputCount) ||
                !CudaKernelOps.TryGetContiguousFloat(output, out var os, out var yp, out int outputCount) ||
                !ReferenceEquals(xs.AllocatorImpl, _allocator) || !ReferenceEquals(os.AllocatorImpl, _allocator)) return false;
            ulong inputStart = unchecked((ulong)xp.ToInt64()), outputStart = unchecked((ulong)yp.ToInt64());
            if (inputStart < outputStart + (ulong)outputCount * sizeof(float) &&
                outputStart < inputStart + (ulong)inputCount * sizeof(float)) return false;
            _allocator.Context.MakeCurrent();
            if (!CudaKernelOps.TryGetKernels(_allocator, out var kernels)) return false;
            EnsureHalfScratch(ref _weightF16, inner * columns);
            EnsureHalfScratch(ref _activationF16, inputCount);
            using var weightElements = _weightF16.Narrow(0, 0, inner * columns);
            using var weight = weightElements.View(columns, inner);
            if (!TryDequantizeF16(weight, host, type, inner, columns, bytes)) return false;
            xs.EnsureDeviceCurrent();
            if (output.StorageOffset != 0 || output.ElementCount() != output.Storage.ElementCount) os.EnsureDeviceCurrent();
            var ws = (CudaStorage)_weightF16.Storage;
            var act = (CudaStorage)_activationF16.Storage;
            IntPtr wp = ws.DevicePtrAtElement(0), ap = act.DevicePtrAtElement(0);
            kernels.LaunchConvertF32F16(xp, ap, inputCount, _allocator.Stream.Handle);
            act.MarkDeviceModified();
            _allocator.Blas.SetStream(_allocator.Stream.Handle);
            float alpha = 1, beta = 0;
            CublasApi.cublasGemmEx(_allocator.Blas.Handle, CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
                (int)columns, (int)input.Sizes[0], (int)inner, ref alpha,
                wp, CublasApi.CUDA_R_16F, (int)inner, ap, CublasApi.CUDA_R_16F, (int)inner, ref beta,
                yp, CublasApi.CUDA_R_32F, (int)columns, CublasApi.CUBLAS_COMPUTE_32F,
                CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
            os.MarkDeviceModified();
            Interlocked.Increment(ref _f16Executed);
            return true;
        }
    }

    private void EnsureHalfScratch(ref Tensor scratch, long elements)
    {
        if (scratch != null && scratch.ElementCount() >= elements) return;
        var larger = new Tensor(_allocator, DType.Float16, elements);
        if (scratch != null) { _allocator.Stream.Synchronize(); scratch.Dispose(); }
        scratch = larger;
    }

    public void ReleaseWorkspace()
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ReleaseHalfScratch();
        }
    }

    private void ReleaseHalfScratch()
    {
        if (_weightF16 == null && _activationF16 == null) return;
        _allocator.Context.MakeCurrent(); _allocator.Stream.Synchronize();
        _weightF16?.Dispose(); _weightF16 = null;
        _activationF16?.Dispose(); _activationF16 = null;
    }

    /// <summary>Project 2..31 input rows against unchanged GGUF Q4_K/Q5_K/Q6_K
    /// storage. False means no native launch occurred; callers keep their
    /// existing route. A successful call increments ExecutedProjections.</summary>
    public unsafe bool TryLinear(Tensor output, Tensor input, IntPtr host, int type, long inner, long columns, long bytes)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int blockBytes = type switch { 12 => 144, 13 => 176, 14 => 210, _ => 0 };
        if (!IsAvailable || host == IntPtr.Zero || blockBytes == 0 || inner <= 0 || inner > int.MaxValue || inner % 256 != 0 ||
            columns <= 0 || columns > int.MaxValue || bytes != checked(inner / 256 * columns * blockBytes) ||
            !CudaKernelOps.TryGetContiguousFloat(output, out var os, out var yp, out int outputCount) ||
            !CudaKernelOps.TryGetContiguousFloat(input, out var xs, out var xp, out int inputCount) ||
            input.DimensionCount != 2 || output.DimensionCount != 2 || input.Sizes[1] != inner || output.Sizes[1] != columns ||
            input.Sizes[0] < 2 || input.Sizes[0] > 31 || output.Sizes[0] != input.Sizes[0] ||
            !ReferenceEquals(os.AllocatorImpl, _allocator) || !ReferenceEquals(xs.AllocatorImpl, _allocator)) return false;
        // Keep the configured prefill route's precision policy. This module
        // replaces only scalar Float32 projections, never an eligible F16 GEMM.
        if (input.Sizes[0] >= CudaQuantizedOps.F16GemmMinRows &&
            2L * inner * columns <= CudaQuantizedOps.F16GemmMaxWeightBytes) return false;
        ulong inputStart = unchecked((ulong)xp.ToInt64()), outputStart = unchecked((ulong)yp.ToInt64());
        if (inputStart < outputStart + (ulong)outputCount * sizeof(float) &&
            outputStart < inputStart + (ulong)inputCount * sizeof(float)) return false;
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            _allocator.Context.MakeCurrent();
            CudaQuantizedOps.PreloadQuantizedWeight(_allocator, host, host, type, inner, columns, bytes);
            if (!CudaQuantizedOps.TryGetResidentDevicePtr(_allocator, host, out var weight, out int storedType) || storedType != type) return false;
            xs.EnsureDeviceCurrent();
            // A contiguous output can be a slice of a larger host-initialized
            // tensor; preserve the elements outside the slice before writing.
            if (output.StorageOffset != 0 || output.ElementCount() != output.Storage.ElementCount)
                os.EnsureDeviceCurrent();
            int rows = checked((int)input.Sizes[0]), width = checked((int)inner), count = checked((int)columns);
            int tile = rows <= 8 ? 8 : 16;
            void** args = stackalloc void*[7] { &weight, &xp, &yp, &type, &width, &count, &rows };
            CudaDriverApi.cuLaunchKernel(tile == 8 ? _tile8 : _tile16,
                checked((uint)((count + 7) / 8)), checked((uint)((rows + tile - 1) / tile)), 1,
                256, 1, 1, 0, _allocator.Stream.Handle, (IntPtr)args, IntPtr.Zero).ThrowOnError();
            os.MarkDeviceModified();
            Interlocked.Increment(ref _executed);
            return true;
        }
    }

    public void Dispose()
    {
        lock (_sync)
        {
            if (_disposed) return;
            _disposed = true;
            ReleaseHalfScratch();
            if (_module == null) return;
            _allocator.Context.MakeCurrent();
            _allocator.Stream.Synchronize();
            _module.Dispose(); _module = null;
        }
    }
}
