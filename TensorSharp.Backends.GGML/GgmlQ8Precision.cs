// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML;

/// <summary>Weight-scoped CUDA Q8_0 projections that retain F32 activations.
/// Keys are reference counted and must be retired before their storage is freed.
/// This policy does not expand quantized weights into a persistent F32 copy.</summary>
public static class GgmlQ8Precision
{
    public static void RegisterWeight(IntPtr key)
    {
        if (key == IntPtr.Zero) throw new ArgumentException("A live weight key is required.", nameof(key));
        if (GgmlNative.TSGgml_RegisterQ8F32Weight(key) != 1)
            throw new InvalidOperationException("Native Q8/F32 precision registration failed.");
    }

    public static void UnregisterWeight(IntPtr key)
        => GgmlNative.TSGgml_UnregisterQ8F32Weight(key);
}

internal static partial class GgmlNative
{
    [LibraryImport(DllName)]
    internal static partial int TSGgml_RegisterQ8F32Weight(IntPtr key);

    [LibraryImport(DllName)]
    internal static partial void TSGgml_UnregisterQ8F32Weight(IntPtr key);
}
