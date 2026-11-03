// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;
internal sealed unsafe partial class CudaKernels
{
    private IntPtr w8HeadBf16ToF32;
    internal const string W8HeadBitBridgeSymbol = "ts_h3_weight_bf16_f32";
    internal static IntPtr ResolveW8HeadBitBridge(Func<string, IntPtr> find, Action<IntPtr> load)
    {
        IntPtr function = find(W8HeadBitBridgeSymbol);
        if (function == IntPtr.Zero) throw new NotSupportedException("The existing BF16-to-F32 bit bridge is unavailable.");
        load(function);
        return function;
    }
    internal void EnsureW8HeadBitBridge()
    {
        if (w8HeadBf16ToF32 != IntPtr.Zero) return;
        w8HeadBf16ToF32 = ResolveW8HeadBitBridge(module.GetFunction, static f => CudaDriverApi.cuFuncLoad(f).ThrowOnError());
    }
    internal void LaunchW8HeadBitBridge(IntPtr input, IntPtr output, int elements, IntPtr stream)
    {
        if (elements <= 0 || input == IntPtr.Zero || output == IntPtr.Zero)
            throw new ArgumentException("Positive BF16/F32 head buffers are required.");
        EnsureW8HeadBitBridge();
        void** args = stackalloc void*[3] { &input, &output, &elements };
        Launch(w8HeadBf16ToF32, Grid(elements), 1, 1, BlockSize, 1, 1, 0, stream, args);
    }
}
