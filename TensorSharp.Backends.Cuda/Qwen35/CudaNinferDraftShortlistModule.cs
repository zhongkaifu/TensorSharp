// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#nullable enable
using System;
using System.IO;
using System.Text.RegularExpressions;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

/// <summary>Explicit original Q4G64_F16S shortlist head. Actual BF16 input/output,
/// original raw adjacent signed-nibble codes and row-major F16 G64 scales.
/// Caller owns complete required device spans and stream completion; pointer-span
/// checks are not independent proofs of CUDA allocation capacity. No serving hook.</summary>
internal sealed unsafe class CudaNinferDraftShortlistModule : ICudaNinferDraftShortlistKernels
{
    internal const int HeadRows = 131072, Columns = 5120, MaximumRows = 33;
    internal const long CodeBytes = 335544320, ScaleBytes = 20971520, WeightBytes = 356515840;
    internal const string NumericalPolicy = "original-q4g64-f16s-shortlist-131072x5120-e600-v1";
    private readonly CudaModule _module;
    private readonly IntPtr _context, _remap;
    private readonly IntPtr[] _projection = new IntPtr[9];
    private bool _disposed;
    public string ImageSha256 => _module.ImageSha256;

    internal CudaNinferDraftShortlistModule(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        CudaDriverApi.cuCtxGetCurrent(out _context).ThrowOnError();
        if (_context == IntPtr.Zero) throw new InvalidOperationException("Make the owning CUDA context current before loading the shortlist module.");
        _module = CudaModule.LoadFromFile(Path.GetFullPath(path), identify: true);
        try
        {
            var symbols = _module.EnumerateFunctions();
            if (symbols.Length != 10) throw new InvalidOperationException("Shortlist requires exactly nine original projection kernels plus remap.");
            IntPtr remap = IntPtr.Zero;
            foreach (var symbol in symbols)
            {
                if (symbol.Function == IntPtr.Zero) throw new InvalidOperationException("Null shortlist kernel binding.");
                int index = ClassifyProjectionSymbol(symbol.Name);
                if (index >= 0)
                {
                    if (_projection[index] != IntPtr.Zero) throw new InvalidOperationException("Duplicate original shortlist projection binding.");
                    _projection[index] = symbol.Function;
                }
                else if (symbol.Name == "ts_ninfer_shortlist_remap" && remap == IntPtr.Zero) remap = symbol.Function;
                else throw new InvalidOperationException("Unexpected shortlist kernel: " + symbol.Name);
            }
            for (int i = 0; i < _projection.Length; i++)
            {
                if (_projection[i] == IntPtr.Zero) throw new InvalidOperationException("Missing original shortlist route.");
                CudaDriverApi.cuFuncLoad(_projection[i]).ThrowOnError();
                if (i == 0) CheckParameters(_projection[i], new ulong[] { 0, 8, 16, 24, 32, 40, 44, 48 }, new ulong[] { 8, 8, 8, 8, 8, 4, 4, 1 });
                else if (i < 8) CheckParameters(_projection[i], new ulong[] { 0, 8, 16, 24, 32, 33 }, new ulong[] { 8, 8, 8, 8, 1, 1 });
                else CheckParameters(_projection[i], new ulong[] { 0, 8, 16, 24, 32, 36, 40, 44 }, new ulong[] { 8, 8, 8, 8, 4, 4, 4, 4 });
            }
            if (remap == IntPtr.Zero) throw new InvalidOperationException("Missing original shortlist remap bridge.");
            _remap = remap; CudaDriverApi.cuFuncLoad(_remap).ThrowOnError();
            CheckParameters(_remap, new ulong[] { 0, 8, 16, 24 }, new ulong[] { 8, 4, 8, 4 });
        }
        catch { _module.Dispose(); throw; }
    }

    internal static int ClassifyProjectionSymbol(string name)
    {
        if (name.StartsWith("_ZN6ninfer3ops6detail", StringComparison.Ordinal))
        {
            if (name.Contains("q4_rowsplit_gemv_kernel", StringComparison.Ordinal)
                && name.Contains("Q4RowSplitGemvScheduleILi4ELi1ELi16ELi1E", StringComparison.Ordinal)) return 0;
            if (name.Contains("q4_small_t_mma_kernel", StringComparison.Ordinal))
            {
                var match = Regex.Match(name, @"Q4DraftHeadGeometryILi5120EEELi8ELi([2-8])E", RegexOptions.CultureInvariant);
                if (match.Success) return match.Groups[1].Value[0] - '1';
            }
            if (name.Contains("q4_rowsplit_gemm_mma_kernel", StringComparison.Ordinal)
                && name.Contains("Q4RowSplitMmaGemmScheduleILi64ELi128ELi64ELi64ELi32ELi2ELi1E", StringComparison.Ordinal)
                && name.Contains("Lb0EE", StringComparison.Ordinal)) return 8;
        }
        return -1;
    }
    private static void CheckParameters(IntPtr kernel, ulong[] offsets, ulong[] sizes)
    {
        for (int i = 0; i < offsets.Length; i++)
        {
            CudaDriverApi.cuFuncGetParamInfo(kernel, (UIntPtr)(uint)i, out var offset, out var size).ThrowOnError();
            if (offset.ToUInt64() != offsets[i] || size.ToUInt64() != sizes[i]) throw new InvalidOperationException("Original shortlist parameter ABI changed at" + i + ".");
        }
        if (CudaDriverApi.cuFuncGetParamInfo(kernel, (UIntPtr)(uint)offsets.Length, out _, out _) != 1)
            throw new InvalidOperationException("Unexpected extra shortlist kernel parameter.");
    }

    internal readonly record struct ProjectionPlan(int Index, int Threads, int Blocks, long InputBytes, long OutputBytes, string Route);
    internal static ProjectionPlan Plan(int rows)
    {
        if (rows < 1 || rows > MaximumRows) throw new ArgumentOutOfRangeException(nameof(rows), "Only the original shortlist T1..33 is supported; no alternate route or chunking.");
        return rows == 1 ? new(0, 128, HeadRows / 4, Columns * 2L, HeadRows * 2L, "r4-w1-direct-gemv")
            : rows <= 8 ? new(rows - 1, 256, HeadRows / 16, rows * Columns * 2L, rows * HeadRows * 2L, "exact-smallT8")
            : new(8, 128, HeadRows / 64, rows * Columns * 2L, rows * HeadRows * 2L, "r64-c128-mma-boundary");
    }
    public void Project(IntPtr codes, IntPtr scales, IntPtr inputBf16, IntPtr outputBf16, int rows, IntPtr stream)
    {
        var p = Plan(rows); CheckContext();
        CheckPointer(codes, CodeBytes, 16); CheckPointer(scales, ScaleBytes, 16);
        CheckPointer(inputBf16, p.InputBytes, 16); CheckPointer(outputBf16, p.OutputBytes, 16);
        Separate(codes, CodeBytes, scales, ScaleBytes); Separate(codes, CodeBytes, inputBf16, p.InputBytes);
        Separate(scales, ScaleBytes, inputBf16, p.InputBytes); Separate(codes, CodeBytes, outputBf16, p.OutputBytes);
        Separate(scales, ScaleBytes, outputBf16, p.OutputBytes); Separate(inputBf16, p.InputBytes, outputBf16, p.OutputBytes);
        byte epilogue = 0, rowPolicy = 0; int headRows = HeadRows, columns = Columns, padded = Columns; IntPtr tail = IntPtr.Zero;
        if (rows == 1)
        {
            void** args = stackalloc void*[] { &inputBf16, &codes, &scales, &outputBf16, &tail, &headRows, &columns, &epilogue };
            Launch(_projection[0], p.Blocks, p.Threads, stream, args);
        }
        else if (rows <= 8)
        {
            void** args = stackalloc void*[] { &inputBf16, &codes, &scales, &outputBf16, &epilogue, &rowPolicy };
            Launch(_projection[p.Index], p.Blocks, p.Threads, stream, args);
        }
        else
        {
            void** args = stackalloc void*[] { &inputBf16, &codes, &scales, &outputBf16, &headRows, &columns, &rows, &padded };
            Launch(_projection[8], p.Blocks, p.Threads, stream, args);
        }
    }
    public void Remap(IntPtr candidateIds, IntPtr map, long count, IntPtr stream)
    {
        if (count < 1 || count > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(count));
        CheckContext(); long bytes = count * 4; CheckPointer(candidateIds, bytes, 4); CheckPointer(map, HeadRows * 4L, 4); Separate(candidateIds, bytes, map, HeadRows * 4L);
        int n = HeadRows, originalCount = (int)count;
        void** args = stackalloc void*[] { &candidateIds, &originalCount, &map, &n };
        Launch(_remap, checked((int)((count + 255) / 256)), 256, stream, args);
    }
    private void CheckContext()
    {
        ObjectDisposedException.ThrowIf(_disposed, this); CudaDriverApi.cuCtxGetCurrent(out var actual).ThrowOnError();
        if (actual != _context) throw new InvalidOperationException("The shortlist module's owning CUDA context is not current.");
    }
    private static void CheckPointer(IntPtr pointer, long bytes, int alignment)
    {
        if (pointer.ToInt64() <= 0 || pointer.ToInt64() % alignment != 0 || bytes < 1 || pointer.ToInt64() > long.MaxValue - bytes)
            throw new ArgumentException("Original shortlist device span/alignment is invalid.");
    }
    private static void Separate(IntPtr a, long ab, IntPtr b, long bb)
    {
        if (a.ToInt64() < b.ToInt64() + bb && b.ToInt64() < a.ToInt64() + ab) throw new ArgumentException("Original shortlist live spans overlap.");
    }
    private static void Launch(IntPtr kernel, int blocks, int threads, IntPtr stream, void** args) =>
        CudaDriverApi.cuLaunchKernel(kernel, (uint)blocks, 1, 1, (uint)threads, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
    public void Dispose() { if (_disposed) return; _disposed = true; _module.Dispose(); }
}
