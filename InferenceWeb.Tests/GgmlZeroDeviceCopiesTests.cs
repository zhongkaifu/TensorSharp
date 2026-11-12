// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

/// <summary>
/// Device copies primed together (<see cref="GgmlBasicOps.ZeroDeviceCopies"/>) are slices of one shared device buffer: each
/// reads back as zeros, a range written into one lands in that slice only, the per-key paths (range uploads and downloads,
/// a device-side zero fill) address each slice at its offset, and every key releases its copy on its own. Only backends that
/// keep separate device copies (CUDA, Vulkan) place them; elsewhere the call primes nothing.
/// </summary>
public sealed class GgmlZeroDeviceCopiesTests
{
    [GgmlFact(BackendType.GgmlCuda)]
    public void CudaSlicesOfASharedBufferBehaveAsSeparateCopies() => Verify(GgmlBackendType.Cuda);

    [GgmlFact(BackendType.GgmlVulkan)]
    public void VulkanSlicesOfASharedBufferBehaveAsSeparateCopies() => Verify(GgmlBackendType.Vulkan);

    private static void Verify(GgmlBackendType backend)
    {
        _ = new GgmlAllocator(new GgmlContext([0], backend), 0);
        long[] sizes = { 4096, 1000, 65536 + 12, 4 };
        var keys = new IntPtr[sizes.Length];
        try
        {
            for (int i = 0; i < keys.Length; i++)
            {
                keys[i] = Marshal.AllocHGlobal((IntPtr)sizes[i]);
                Fill(keys[i], sizes[i], 0xAB);
            }

            Assert.True(GgmlBasicOps.ZeroDeviceCopies(keys, sizes));
            for (int i = 0; i < keys.Length; i++)
            {
                Assert.Equal(sizes[i], GgmlBasicOps.CachedBufferBytes(keys[i]));
                GgmlBasicOps.SyncHostBufferRanges(keys[i], new[] { 0L }, new[] { sizes[i] });
                AssertBytes(keys[i], sizes[i], _ => 0, $"key {i} after priming");
            }

            // A range of key 1 uploaded to its slice, and key 2's whole bytes to its own: neither reaches the other.
            for (int b = 0; b < 300; b++) Marshal.WriteByte(keys[1], 100 + b, (byte)(b + 1));
            GgmlBasicOps.UploadHostBufferRanges(keys[1], new[] { 100L }, new[] { 300L });
            Fill(keys[2], sizes[2], 0x5A);
            GgmlBasicOps.UploadHostBufferRanges(keys[2], new[] { 0L }, new[] { sizes[2] });
            for (int i = 0; i < keys.Length; i++) Fill(keys[i], sizes[i], 0xEE);
            for (int i = 0; i < keys.Length; i++) GgmlBasicOps.SyncHostBufferRanges(keys[i], new[] { 0L }, new[] { sizes[i] });
            AssertBytes(keys[0], sizes[0], _ => 0, "key 0 beside an upload");
            AssertBytes(keys[1], sizes[1], o => o is >= 100 and < 400 ? (byte)(o - 99) : (byte)0, "key 1's uploaded range");
            AssertBytes(keys[2], sizes[2], _ => 0x5A, "key 2's upload");
            AssertBytes(keys[3], sizes[3], _ => 0, "key 3 beside an upload");

            // The per-key device-side zero fill of key 1 (already primed: a slice) clears that slice only. That path serves
            // one-device processes; one that set up several devices (a tensor-parallel test before this) refuses it.
            if (PerKeyFillServed())
            {
                Assert.True(GgmlBasicOps.ZeroDeviceCopies(new[] { keys[1] }, new[] { sizes[1] }));
                for (int i = 0; i < keys.Length; i++) Fill(keys[i], sizes[i], 0xEE);
                GgmlBasicOps.SyncHostBufferRanges(keys[1], new[] { 0L }, new[] { sizes[1] });
                GgmlBasicOps.SyncHostBufferRanges(keys[2], new[] { 0L }, new[] { sizes[2] });
                AssertBytes(keys[1], sizes[1], _ => 0, "key 1 zero-filled again");
                AssertBytes(keys[2], sizes[2], _ => 0x5A, "key 2 beside a zero fill");
            }

            // Released one at a time, the other slices stay readable until their own release.
            GgmlBasicOps.InvalidateHostBuffer(keys[0]);
            Assert.Equal(0, GgmlBasicOps.CachedBufferBytes(keys[0]));
            Fill(keys[2], sizes[2], 0xEE);
            GgmlBasicOps.SyncHostBufferRanges(keys[2], new[] { 0L }, new[] { sizes[2] });
            AssertBytes(keys[2], sizes[2], _ => 0x5A, "key 2 after key 0's release");
        }
        finally
        {
            foreach (IntPtr key in keys)
            {
                if (key == IntPtr.Zero) continue;
                GgmlBasicOps.InvalidateHostBuffer(key);
                Assert.Equal(0, GgmlBasicOps.CachedBufferBytes(key));
                Marshal.FreeHGlobal(key);
            }
        }
    }

    private static bool PerKeyFillServed()
    {
        IntPtr probe = Marshal.AllocHGlobal(64);
        try { return GgmlBasicOps.ZeroDeviceCopy(probe, 64); }
        finally
        {
            GgmlBasicOps.InvalidateHostBuffer(probe);
            Marshal.FreeHGlobal(probe);
        }
    }

    private static void Fill(IntPtr p, long bytes, byte value)
    {
        for (long o = 0; o < bytes; o++) Marshal.WriteByte(p, (int)o, value);
    }

    private static void AssertBytes(IntPtr p, long bytes, Func<long, byte> expected, string what)
    {
        for (long o = 0; o < bytes; o++)
        {
            byte got = Marshal.ReadByte(p, (int)o);
            if (got != expected(o))
                Assert.Fail($"{what}: byte {o} is 0x{got:X2}, expected 0x{expected(o):X2}");
        }
    }
}
