// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#nullable enable
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

/// <summary>Owned placement-only adapters for the explicit raw-resident FI body.
/// Input compaction uses RN BF16. Row scatter preserves F32 bits and leaves the
/// target's existing rounding/residual/SwiGLU boundaries to its caller.</summary>
internal sealed unsafe class CudaNinferFlashInferBridge : IDisposable
{
    internal const int MaximumRows = 8, MaximumColumns = 17408, MaximumOutputRows = 34816;
    internal const long MaximumCastElements = (long)MaximumRows * MaximumColumns;
    internal const string CastSymbol = "ts_ninfer_fi_cast_bf16", ScatterSymbol = "ts_ninfer_fi_scatter_f32";
    private readonly CudaModule _module;
    private readonly IntPtr _cast, _scatter, _context;
    private bool _disposed;
    internal string ImageSha256 => _module.ImageSha256;

    internal CudaNinferFlashInferBridge(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        CudaDriverApi.cuCtxGetCurrent(out _context).ThrowOnError();
        if (_context == IntPtr.Zero || IntPtr.Size != 8)
            throw new PlatformNotSupportedException("The FI bridge requires its current 64-bit CUDA context.");
        _module = CudaModule.LoadFromFile(path, identify: true);
        try
        {
            var functions = _module.EnumerateFunctions();
            if (functions.Length != 2) throw new InvalidOperationException("The FI bridge requires exactly its two owned kernels.");
            _cast = Bind(CastSymbol, new ulong[] { 0, 8, 16 }, new ulong[] { 8, 8, 8 });
            _scatter = Bind(ScatterSymbol, new ulong[] { 0, 8, 16, 20, 24 }, new ulong[] { 8, 8, 4, 4, 4 });
        }
        catch { _module.Dispose(); throw; }
    }

    private IntPtr Bind(string symbol, ulong[] offsets, ulong[] sizes)
    {
        IntPtr kernel = _module.GetFunction(symbol);
        CudaDriverApi.cuFuncLoad(kernel).ThrowOnError();
        for (uint i = 0; i < offsets.Length; i++)
        {
            CudaDriverApi.cuFuncGetParamInfo(kernel, (UIntPtr)i, out var offset, out var size).ThrowOnError();
            if (offset.ToUInt64() != offsets[i] || size.ToUInt64() != sizes[i])
                throw new InvalidOperationException("The FI bridge device ABI differs from the owned adapter.");
        }
        if (CudaDriverApi.cuFuncGetParamInfo(kernel, (UIntPtr)(uint)offsets.Length, out _, out _) != 1)
            throw new InvalidOperationException("The FI bridge has an unexpected extra argument.");
        return kernel;
    }

    internal static long ValidateCast(long count)
    {
        if (count < 1 || count > MaximumCastElements) throw new ArgumentOutOfRangeException(nameof(count));
        return checked(count * sizeof(float));
    }
    internal static long ValidateScatter(int outputRows, int rows, CudaNinferOriginalRowMapping mapping)
    {
        if (rows < 1 || rows > MaximumRows || outputRows < 1 || outputRows > MaximumOutputRows || outputRows % 128 != 0)
            throw new ArgumentOutOfRangeException(nameof(rows));
        if (mapping == CudaNinferOriginalRowMapping.GateUpRows8 && outputRows != 34816
            || mapping == CudaNinferOriginalRowMapping.AttentionQueryGateKeyValue && outputRows != 14336)
            throw new ArgumentException("The FI bridge only accepts the fixed target row mappings.");
        if (mapping is not (CudaNinferOriginalRowMapping.Identity or CudaNinferOriginalRowMapping.GateUpRows8
            or CudaNinferOriginalRowMapping.AttentionQueryGateKeyValue))
            throw new ArgumentOutOfRangeException(nameof(mapping));
        return checked((long)rows * outputRows * sizeof(float));
    }
    internal static void ValidateSpans(IntPtr input, long inputBytes, int inputAlignment,
        IntPtr output, long outputBytes, int outputAlignment)
    {
        long a = input.ToInt64(), b = output.ToInt64();
        if (inputBytes < 1 || outputBytes < 1 || inputAlignment < 1 || outputAlignment < 1
            || a <= 0 || b <= 0 || a % inputAlignment != 0 || b % outputAlignment != 0
            || a > long.MaxValue - inputBytes || b > long.MaxValue - outputBytes)
            throw new ArgumentException("The FI bridge spans are invalid, unaligned or overflowing.");
        if (a < b + outputBytes && b < a + inputBytes)
            throw new ArgumentException("The FI bridge cannot overwrite simultaneously live input bytes.");
    }
    private void CheckContext()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        CudaDriverApi.cuCtxGetCurrent(out var current).ThrowOnError();
        if (current != _context) throw new InvalidOperationException("Make the FI bridge's owning CUDA context current.");
    }
    internal void Cast(IntPtr inputF32, IntPtr outputBf16, long count, IntPtr stream)
    {
        long bytes = ValidateCast(count);
        ValidateSpans(inputF32, bytes, 4, outputBf16, count * 2, 2); CheckContext();
        ulong elements = checked((ulong)count);
        void** args = stackalloc void*[] { &inputF32, &outputBf16, &elements };
        CudaDriverApi.cuLaunchKernel(_cast, checked((uint)((count + 255) / 256)), 1, 1, 256, 1, 1, 0,
            stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
    }
    internal void Scatter(IntPtr inputOriginalF32, IntPtr outputTsF32, int outputRows, int rows,
        CudaNinferOriginalRowMapping mapping, IntPtr stream)
    {
        long bytes = ValidateScatter(outputRows, rows, mapping);
        ValidateSpans(inputOriginalF32, bytes, 4, outputTsF32, bytes, 4); CheckContext();
        int rowMapping = (int)mapping;
        void** args = stackalloc void*[] { &inputOriginalF32, &outputTsF32, &outputRows, &rows, &rowMapping };
        CudaDriverApi.cuLaunchKernel(_scatter, checked((uint)((bytes / 4 + 255) / 256)), 1, 1, 256, 1, 1, 0,
            stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
    }
    public void Dispose()
    {
        if (_disposed) return;
        CheckContext(); _module.Dispose(); _disposed = true;
    }
}
