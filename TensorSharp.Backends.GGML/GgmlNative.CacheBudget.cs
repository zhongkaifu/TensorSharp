// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML;

internal static partial class GgmlNative
{
    [LibraryImport(DllName, EntryPoint = "TSGgml_AttachSharedCacheBudget")]
    internal static partial int AttachSharedCacheBudget(IntPtr context, IntPtr reserve, IntPtr commit, IntPtr release);

    [LibraryImport(DllName, EntryPoint = "TSGgml_AttachSharedCacheBudgetEx")]
    internal static partial int AttachSharedCacheBudgetEx(IntPtr context, IntPtr reserve, IntPtr commit, IntPtr release, int includeGraphBuffers);

    [LibraryImport(DllName, EntryPoint = "TSGgml_DetachSharedCacheBudget")]
    internal static partial int DetachSharedCacheBudget(IntPtr context);
}
