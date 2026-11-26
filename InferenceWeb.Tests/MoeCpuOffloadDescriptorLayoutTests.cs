// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// Layout contract for the whole-model decode descriptors that carry the MoE
/// CPU-offload flag (<c>--n-cpu-moe</c>). Each of these is passed by pointer to a
/// native kernel whose C++ struct must agree field for field.
///
/// The kernels do check <c>struct_bytes</c> at entry, but that handshake only
/// catches a size change. Appending <c>CpuMoe</c> in the wrong place — or a later
/// edit reordering the int32 run — keeps the size identical while silently
/// shifting every field after it, and the failure mode is garbage output rather
/// than a clean rejection. These tests pin the shape the native side assumes:
/// pointers, then int64, then int32 (with <c>CpuMoe</c> last in that run), then
/// float.
///
/// Mirrors <see cref="Qwen35TokenDecodeContractTests"/>, which covers the
/// Qwen3.5/3.6 descriptor the same way.
/// </summary>
public class MoeCpuOffloadDescriptorLayoutTests
{
    private static long Align(long offset, long alignment)
        => ((offset + alignment - 1) / alignment) * alignment;

    /// <summary>
    /// Assert that <typeparamref name="T"/> is exactly
    /// <paramref name="pointers"/> IntPtr, <paramref name="int64s"/> long,
    /// <paramref name="int32s"/> int and <paramref name="floats"/> float — in
    /// that order — and that <c>CpuMoe</c> is the last int32.
    /// </summary>
    private static void AssertDescriptorLayout<T>(
        int pointers, int int64s, int int32s, int floats,
        string firstPointerField, string firstInt32Field,
        int trailingPointers = 0, string firstTrailingPointerField = null) where T : struct
    {
        FieldInfo[] fields = typeof(T).GetFields(BindingFlags.Public | BindingFlags.Instance);

        Assert.Equal(pointers + trailingPointers, fields.Count(f => f.FieldType == typeof(IntPtr)));
        Assert.Equal(int64s, fields.Count(f => f.FieldType == typeof(long)));
        Assert.Equal(int32s, fields.Count(f => f.FieldType == typeof(int)));
        Assert.Equal(floats, fields.Count(f => f.FieldType == typeof(float)));
        // Anything else in the struct (bool, enum, a nested type) would marshal
        // differently than the native side expects.
        Assert.All(fields, f => Assert.True(
            f.FieldType == typeof(IntPtr) || f.FieldType == typeof(long) ||
            f.FieldType == typeof(int) || f.FieldType == typeof(float),
            $"unexpected field type {f.FieldType} on {typeof(T).Name}.{f.Name}"));

        long int64Start = Align((long)pointers * IntPtr.Size, sizeof(long));
        long int32Start = int64Start + (long)int64s * sizeof(long);
        // int32 and float are both 4 bytes and pack into one run. A descriptor
        // may append an extra pointer run at the very end (e.g. the optional
        // F16 prefill-weight copies on the GPT-OSS descriptor) so every
        // pre-existing field keeps its offset.
        long floatEnd = int32Start + ((long)int32s + floats) * sizeof(int);
        long trailStart = Align(floatEnd, IntPtr.Size);
        long expectedSize = Align(
            trailingPointers > 0 ? trailStart + (long)trailingPointers * IntPtr.Size : floatEnd,
            Math.Max(IntPtr.Size, sizeof(long)));

        Assert.Equal(0, Marshal.OffsetOf<T>(firstPointerField).ToInt64());
        Assert.Equal(int32Start, Marshal.OffsetOf<T>(firstInt32Field).ToInt64());
        if (trailingPointers > 0 && firstTrailingPointerField != null)
            Assert.Equal(trailStart, Marshal.OffsetOf<T>(firstTrailingPointerField).ToInt64());
        Assert.Equal(expectedSize, Marshal.SizeOf<T>());

        // CpuMoe is appended at the end of the int32 run so every field before it
        // keeps its offset; the native kernels read it there.
        Assert.Equal(
            int32Start + ((long)int32s - 1) * sizeof(int),
            Marshal.OffsetOf<T>("CpuMoe").ToInt64());
    }

    [Fact]
    public void GptOssLayerDecodeArgs_MatchesNativeLayout()
        => AssertDescriptorLayout<TensorSharp.GGML.GptOssLayerDecodeArgs>(
            pointers: 21, int64s: 21, int32s: 22, floats: 5,
            firstPointerField: "AttnNormW", firstInt32Field: "StructBytes",
            trailingPointers: 7, firstTrailingPointerField: "QkvWF16");
}
