// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.IO;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

/// <summary>Original Ignis BF16 T8 cores with the owned, unrounded F32 output policy.
/// BF16 input conversion and each consumer's rounding remain the caller's responsibility.</summary>
internal sealed unsafe class CudaNinferBf16MmaModule : IDisposable
{
    internal const string ArithmeticPolicy = "ignis-e600-bf16-t8-original-mma-f32-store-v1";
    internal const long ScratchBytes = 8L * 6144 * 2;
    internal const int Threads = 256, DynamicSharedBytes = 98304;
    private readonly CudaModule _module;
    private readonly IntPtr _context;
    private readonly IntPtr[] _projection = new IntPtr[2];
    private bool _disposed;
    internal string ImageSha256 => _module.ImageSha256;

    internal CudaNinferBf16MmaModule(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        CudaDriverApi.cuCtxGetCurrent(out _context).ThrowOnError();
        if (_context == IntPtr.Zero) throw new InvalidOperationException("A current owning CUDA context is required.");
        _module = CudaModule.LoadFromFile(Path.GetFullPath(path), identify: true);
        try
        {
            var symbols = _module.EnumerateFunctions();
            if (symbols.Length != 4) throw new InvalidOperationException("BF16 MMA requires the four qualified original-core instantiations.");
            for (int shape = 0; shape < 2; shape++)
                for (int stock = 0; stock < 2; stock++)
                {
                    string name = KernelName(shape == 0 ? 14336 : 5120, shape == 0 ? 5120 : 6144, stock != 0);
                    IntPtr function = IntPtr.Zero;
                    foreach (var symbol in symbols)
                        if (symbol.Name == name)
                        {
                            if (function != IntPtr.Zero) throw new InvalidOperationException("Duplicate BF16 MMA entry.");
                            function = symbol.Function;
                        }
                    if (function == IntPtr.Zero) throw new InvalidOperationException("Missing qualified BF16 MMA entry.");
                    CudaDriverApi.cuFuncLoad(function).ThrowOnError();
                    for (uint i = 0; i < 4; i++)
                    {
                        CudaDriverApi.cuFuncGetParamInfo(function, (UIntPtr)i, out var offset, out var size).ThrowOnError();
                        ValidateParameterStorage(i, stock != 0, offset.ToUInt64(), size.ToUInt64());
                    }
                    if (CudaDriverApi.cuFuncGetParamInfo(function, (UIntPtr)4, out _, out _) != 1)
                        throw new InvalidOperationException("Unexpected BF16 MMA argument.");
                    // This fixed S3 schedule needs 96KiB; set the opt-in before
                    // any capture or publication of a runnable projection.
                    CudaDriverApi.cuFuncSetAttribute(function,
                        CudaDriverApi.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, DynamicSharedBytes).ThrowOnError();
                    if (stock == 0) _projection[shape] = function;
                }
        }
        catch { _module.Dispose(); throw; }
    }

    internal static string KernelName(int outputRows, int columns, bool stock = false)
    {
        _ = Plan(outputRows, columns, 8);
        string prefix = "_ZN6ninfer3ops6detail20bf16_gemm_mma_kernelINS1_16Bf16GemvGeometryILi"
            + outputRows + "ELi" + columns + "EEENS1_15Bf16MmaScheduleILi32ELi32ELi256ELi16ELi8ELi3ELi1ELNS0_5CacheE1ELS6_1ELNS1_23Bf16MmaFragmentPipelineE1ELNS1_13Bf16MmaRasterE0ELNS1_14Bf16MmaSwizzleE1ELi1EEELb0E";
        return prefix + (stock ? "NS1_23Bf16MmaContiguousOutputEEEvPK13__nv_bfloat16SE_T2_i"
            : "N11tensorsharp15ninfer_bf16_mma24TsNinferBf16MmaF32OutputILi" + outputRows + "EEEEEvPK13__nv_bfloat16SH_T2_i");
    }

    internal static void ValidateParameterStorage(uint index, bool stock, ulong offset, ulong size)
    {
        ulong expectedOffset = index switch { 0 => 0, 1 => 8, 2 => 16, 3 => stock ? 32UL : 24UL,
            _ => throw new ArgumentOutOfRangeException(nameof(index)) };
        ulong expectedSize = index == 3 ? 4UL : index == 2 && stock ? 16UL : 8UL;
        if (offset != expectedOffset || size != expectedSize)
            throw new InvalidOperationException("BF16 MMA device ABI differs.");
    }

    internal readonly record struct Layout(int Index, int Blocks, long InputBytes, long WeightBytes, long OutputBytes);
    internal static Layout Plan(int outputRows, int columns, int rows)
    {
        int index = (outputRows, columns) switch { (14336, 5120) => 0, (5120, 6144) => 1, _ => -1 };
        if (index < 0 || rows != 8) throw new ArgumentOutOfRangeException(nameof(rows), "Only the two qualified BF16 T8 exception shapes are supported.");
        return new(index, outputRows / 32, 8L * columns * 2, (long)outputRows * columns * 2, 8L * outputRows * 4);
    }

    internal static void ValidateBuffers(IntPtr input, IntPtr weights, IntPtr output, Layout layout)
    {
        Pointer(input, layout.InputBytes, 16); Pointer(weights, layout.WeightBytes, 16); Pointer(output, layout.OutputBytes, 4);
        if (Overlap(input, layout.InputBytes, output, layout.OutputBytes)
            || Overlap(weights, layout.WeightBytes, output, layout.OutputBytes))
            throw new ArgumentException("BF16 MMA output must not alias its inputs.");
    }
    private static void Pointer(IntPtr pointer, long bytes, int alignment)
    {
        long address = pointer.ToInt64();
        if (address <= 0 || address % alignment != 0 || bytes <= 0 || address > long.MaxValue - bytes)
            throw new ArgumentException("A nonzero, aligned, nonoverflowing device span is required.");
    }
    private static bool Overlap(IntPtr a, long an, IntPtr b, long bn)
        => a.ToInt64() < b.ToInt64() + bn && b.ToInt64() < a.ToInt64() + an;

    internal void Project(IntPtr inputBf16, IntPtr weights, IntPtr outputF32, int outputRows, int columns, int rows, IntPtr stream)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var plan = Plan(outputRows, columns, rows);
        ValidateBuffers(inputBf16, weights, outputF32, plan);
        CudaDriverApi.cuCtxGetCurrent(out var context).ThrowOnError();
        if (context != _context) throw new InvalidOperationException("BF16 MMA must run on its owning context.");
        // The by-value F32 output policy contains exactly this one device pointer.
        // Exact qualified ordinary launch of the pristine original MMA core.
        void** args = stackalloc void*[] { &inputBf16, &weights, &outputF32, &rows };
        CudaDriverApi.cuLaunchKernel(_projection[plan.Index], (uint)plan.Blocks, 1, 1, Threads, 1, 1, DynamicSharedBytes,
            stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
    }
    public void Dispose() { if (_disposed) return; _disposed = true; _module.Dispose(); }
}
