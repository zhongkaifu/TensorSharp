// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML;

[StructLayout(LayoutKind.Sequential)]
public struct QwenImage21Weight
{
    public IntPtr Data;
    public int Type, Reserved;
    public long Ne0, Ne1, Bytes;
}

[StructLayout(LayoutKind.Sequential)]
public struct QwenImage21Block
{
    public QwenImage21Weight Q, K, V, Out, Gate, Up, Down;
    public IntPtr NormQ, NormK;
}

[StructLayout(LayoutKind.Sequential)]
public struct QwenImage21Segment
{
    public int Start, End, SourceStart, IsImage;
}

/// <summary>A LoRA update of one projection, applied unmerged (TSGQi21Lora):
/// y = RowScale * (W x) + Up (Down x). Down is ggml [In, Rank] (PyTorch lora_A),
/// Up is ggml [Rank, Out] (PyTorch lora_B) with every scale folded in.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct QwenImage21Lora
{
    public IntPtr Down, Up, RowScale;
    /// <summary>GGML type of both factors: 1 = F16, 0 = F32.</summary>
    public int Type, Rank;
    public long In, Out;
}

[StructLayout(LayoutKind.Sequential)]
public struct QwenImage21BlockLora
{
    public QwenImage21Lora Q, K, V, Out, Gate, Up, Down;
}

/// <summary>A LoRA plug-in's changes to the transformer (TSGQi21Adapter).</summary>
[StructLayout(LayoutKind.Sequential)]
public struct QwenImage21Adapter
{
    public int StructBytes, NumLayers;
    public QwenImage21Lora ImageIn, TextIn, TextOut, TimeIn, TimeOut, Modulation, NormOut, ProjOut;
    public IntPtr Blocks;
    /// <summary>Optional per-call replacement of proj_out ([dim, channels]), uploaded as an input.</summary>
    public IntPtr OutputHead;
    public int OutputHeadType, Reserved;
}

[StructLayout(LayoutKind.Sequential)]
public struct QwenImage21ForwardArgs
{
    public IntPtr Images, Text, TimeEmbedding, Cos, Sin, Output;
    public QwenImage21Weight ImageIn, TextIn, TextOut, TimeIn, TimeOut, Modulation, NormOut, ProjOut;
    public IntPtr TextNorm, Blocks, Segments;
    public int StructBytes, Dim, Heads, HeadDim, Channels, TextDim;
    public int ImageSeq, TextSeq, TotalSeq, PrefixSeq, NumLayers, NumSegments;
    public float Eps;
    /// <summary>Nonzero names one request's text/reference prefix; see <see cref="QwenImage21ForwardPath"/>.</summary>
    public ulong PrefixCacheKey;
    public QwenImage21PrefixCacheType PrefixCacheType;
    /// <summary>Ranks the block weights are sharded over (0/1 = none); see QwenImage21ForwardTp.</summary>
    public int TpRanks;
    /// <summary>Optional <see cref="QwenImage21Adapter"/> (a LoRA plug-in), or zero.</summary>
    public IntPtr Adapter;
}

/// <summary>Storage of the prefix KV cache. <see cref="Auto"/> stores what attention reads
/// (F16 for Metal and CUDA flash attention, F32 otherwise), so cached steps match uncached ones.
/// <see cref="Q8_0"/> (K and V) and <see cref="Q8_0V"/> (V only) store the prefix in 8 bits.</summary>
public enum QwenImage21PrefixCacheType
{
    Auto = 0,
    F32 = 1,
    F16 = 2,
    Q8_0 = 3,
    Q8_0V = 4,
}

/// <summary>Which graph produced a Qwen-Image-2.1 prediction.</summary>
public enum QwenImage21ForwardPath
{
    /// <summary>The whole sequence; no cache was requested.</summary>
    Full = 1,
    /// <summary>The whole sequence; the prefix K/V were stored for later steps.</summary>
    Extract = 2,
    /// <summary>Only the target tokens, attending to the stored prefix.</summary>
    Cached = 3,
    /// <summary>The whole sequence; the cache did not fit the device (native logs why).</summary>
    Declined = 4,
}

[StructLayout(LayoutKind.Sequential)]
public struct QwenImage21PrefixCacheInfo
{
    /// <summary>0 = none for the key, 1 = stored, 2 = declined.</summary>
    public int State;
    public int KeyType, ValueType, Tokens;
    public long Bytes;
}

internal static partial class GgmlNative
{
    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static partial int TSGgml_QwenImage21Forward(in QwenImage21ForwardArgs desc);

    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static unsafe partial int TSGgml_QwenImage21ForwardTp(QwenImage21ForwardArgs** descs, int ranks);

    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static partial void TSGgml_QwenImage21ReleasePrefixCache(ulong key);

    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static partial int TSGgml_QwenImage21GetPrefixCacheInfo(ulong key, out QwenImage21PrefixCacheInfo info);

    public static QwenImage21ForwardPath QwenImage21Forward(in QwenImage21ForwardArgs desc)
    {
        int path = TSGgml_QwenImage21Forward(in desc);
        if (path == 0)
            throw new InvalidOperationException(GetLastErrorMessage("Qwen-Image-2.1 native inference failed."));
        return (QwenImage21ForwardPath)path;
    }

    public static unsafe QwenImage21ForwardPath QwenImage21ForwardTp(QwenImage21ForwardArgs[] descs)
    {
        ArgumentNullException.ThrowIfNull(descs);
        fixed (QwenImage21ForwardArgs* first = descs)
        {
            var pointers = stackalloc QwenImage21ForwardArgs*[descs.Length];
            for (int r = 0; r < descs.Length; r++) pointers[r] = first + r;
            int path = TSGgml_QwenImage21ForwardTp(pointers, descs.Length);
            if (path == 0)
                throw new InvalidOperationException(GetLastErrorMessage("Qwen-Image-2.1 tensor-parallel inference failed."));
            return (QwenImage21ForwardPath)path;
        }
    }

    public static void QwenImage21ReleasePrefixCache(ulong key) => TSGgml_QwenImage21ReleasePrefixCache(key);

    public static QwenImage21PrefixCacheInfo QwenImage21GetPrefixCacheInfo(ulong key)
    {
        TSGgml_QwenImage21GetPrefixCacheInfo(key, out var info);
        return info;
    }
}

public partial class GgmlBasicOps
{
    /// <summary>Complete Qwen-Image-2.1 velocity prediction in one resident-weight GGML graph.</summary>
    public static QwenImage21ForwardPath QwenImage21Forward(in QwenImage21ForwardArgs args) => GgmlNative.QwenImage21Forward(in args);

    /// <summary>One prediction over a tensor-parallel group: descriptor r holds rank r's sharded
    /// block weights (whole heads, a slice of the MLP) and the shared inputs. Rank 0 writes the output.</summary>
    public static QwenImage21ForwardPath QwenImage21ForwardTp(QwenImage21ForwardArgs[] descs) => GgmlNative.QwenImage21ForwardTp(descs);

    /// <summary>Frees the stored prefix K/V of one request key; the next forward with the key stores them again.</summary>
    public static void QwenImage21ReleasePrefixCache(ulong key) => GgmlNative.QwenImage21ReleasePrefixCache(key);

    public static QwenImage21PrefixCacheInfo QwenImage21GetPrefixCacheInfo(ulong key) => GgmlNative.QwenImage21GetPrefixCacheInfo(key);
}
