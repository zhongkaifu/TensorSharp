// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class Qwen35ReadonlyKvExtractTests
{
    private const int Heads = 3, Capacity = 17, Width = 64;
    private delegate bool ExtractRows(Tensor cache, int start, int count, Span<byte> destination, out int written);

    // Exercise the real byte-copy routine, including its Storage contract, without
    // making a production-private helper public just for a test.
    private static readonly ExtractRows Extract = typeof(Qwen35Model)
        .GetMethod("CopyAttentionOut", BindingFlags.Static | BindingFlags.NonPublic)!
        .CreateDelegate<ExtractRows>();

    public static IEnumerable<object[]> RowCases()
    {
        foreach (DType dtype in new[] { DType.Float32, DType.Float16, DType.Q8_0, DType.Q4_0 })
            foreach (var (start, count) in new[] { (0, Capacity), (1, 4), (Capacity - 1, 1) })
                yield return new object[] { dtype, start, count };
    }

    [Theory]
    [MemberData(nameof(RowCases))]
    public void Extract_PacksExactRowsOfEveryHead_WithoutQueuingAnUpload(DType dtype, int start, int count)
    {
        var allocator = new CheckoutAllocator();
        using var cache = new Tensor(allocator, dtype, Heads, Capacity, Width);
        var storage = (CheckoutStorage)cache.Storage;
        byte[] seed = Seed(cache);
        storage.Seed(seed);
        long rowBytes = cache.Storage.ByteLength / (Heads * Capacity);
        int length = checked((int)(Heads * count * rowBytes));
        byte[] actual = Enumerable.Repeat((byte)0x9d, length + 13).ToArray();

        Assert.True(Extract(cache, start, count, actual, out int written));
        Assert.Equal(length, written);
        Assert.Equal(Pack(seed, rowBytes, start, count), actual[..length]);
        Assert.All(actual[length..], b => Assert.Equal((byte)0x9d, b));
        Assert.Equal(seed, storage.Bytes());
        Assert.True(storage.HostReadChecks > 0);
        Assert.Equal(1, storage.ReadOnlyCheckouts);
        Assert.Equal(0, storage.WriteCheckouts);
        Assert.False(storage.HostDirty);
        storage.EnsureDeviceCurrent();
        Assert.Equal(0, storage.Uploads);
    }

    [Theory]
    [InlineData(DType.Float32)]
    [InlineData(DType.Float16)]
    [InlineData(DType.Q8_0)]
    [InlineData(DType.Q4_0)]
    public void Extract_PreservesARealPendingHostWrite(DType dtype)
    {
        var allocator = new CheckoutAllocator();
        using var cache = new Tensor(allocator, dtype, Heads, Capacity, Width);
        var storage = (CheckoutStorage)cache.Storage;
        byte[] seed = Seed(cache);
        storage.Seed(seed);
        // A read-only checkout must not clear an earlier genuine mutation.
        Marshal.WriteByte(storage.PtrAtElement(0), 0x71);
        seed[0] = 0x71;
        byte[] actual = new byte[Heads * (int)(cache.Storage.ByteLength / (Heads * Capacity))];

        Assert.True(Extract(cache, 0, 1, actual, out int written));
        Assert.Equal(actual.Length, written);
        Assert.Equal(Pack(seed, cache.Storage.ByteLength / (Heads * Capacity), 0, 1), actual);
        Assert.True(storage.HostDirty);
        Assert.Equal(1, storage.WriteCheckouts);
        Assert.Equal(1, storage.ReadOnlyCheckouts);
        storage.EnsureDeviceCurrent();
        Assert.Equal(1, storage.Uploads);
    }

    [Theory]
    [InlineData(DType.Float32)]
    [InlineData(DType.Float16)]
    [InlineData(DType.Q8_0)]
    [InlineData(DType.Q4_0)]
    public void Extract_WithAShortDestination_RefusesBeforeCheckoutOrWrite(DType dtype)
    {
        var allocator = new CheckoutAllocator();
        using var cache = new Tensor(allocator, dtype, Heads, Capacity, Width);
        var storage = (CheckoutStorage)cache.Storage;
        byte[] seed = Seed(cache);
        storage.Seed(seed);
        int length = checked((int)(Heads * 4 * cache.Storage.ByteLength / (Heads * Capacity)));
        byte[] actual = Enumerable.Repeat((byte)0x9d, length - 1).ToArray();

        Assert.False(Extract(cache, 1, 4, actual, out int written));
        Assert.Equal(0, written);
        Assert.All(actual, b => Assert.Equal((byte)0x9d, b));
        Assert.Equal(seed, storage.Bytes());
        Assert.Equal(0, storage.ReadOnlyCheckouts);
        Assert.Equal(0, storage.WriteCheckouts);
        Assert.False(storage.HostDirty);
    }

    [CudaTheory]
    [InlineData(DType.Float32)]
    [InlineData(DType.Float16)]
    public void CudaExtract_RefreshesDeviceWrites_ThenAppendPreservesAllOtherCacheBits(DType dtype)
    {
        using var allocator = new CudaAllocator();
        using var cache = new Tensor(allocator, dtype, Heads, Capacity, Width);
        byte[] expected = Seed(cache);
        WriteBytes(cache, expected);
        var storage = (CudaStorage)cache.Storage;
        storage.EnsureDeviceCurrent();

        // First append makes the device authoritative. Extract must refresh the
        // host mirror without scheduling an unchanged cache for H2D afterward.
        Append(3, 1);
        byte[] extracted = new byte[checked((int)(Heads * 5 * cache.Storage.ByteLength / (Heads * Capacity)))];
        Assert.True(Extract(cache, 2, 5, extracted, out int written));
        Assert.Equal(extracted.Length, written);
        Assert.Equal(Pack(expected, cache.Storage.ByteLength / (Heads * Capacity), 2, 5), extracted);
        Assert.False(storage.IsHostDirty);

        Append(7, 2);
        Assert.False(storage.IsHostDirty);
        Assert.Equal(expected, ReadBytes(cache));

        // Genuine host edits remain pending even across the same read helper;
        // the following append must upload them rather than silently discard.
        cache.Storage.SetElementAsFloat(0, 8.5f);
        byte[] edit = dtype == DType.Float32 ? BitConverter.GetBytes(8.5f)
            : BitConverter.GetBytes(BitConverter.HalfToUInt16Bits((System.Half)8.5f));
        edit.CopyTo(expected, 0);
        Assert.True(storage.IsHostDirty);
        Assert.True(Extract(cache, 0, 1, new byte[Heads * Width * dtype.Size()], out _));
        Assert.True(storage.IsHostDirty);
        Append(11, 1);
        Assert.Equal(expected, ReadBytes(cache));

        void Append(int start, int count)
        {
            using var source = new Tensor(allocator, DType.Float32, Heads, count, Width);
            float[] values = Enumerable.Range(0, Heads * count * Width).Select(i => (i + start) / 64f).ToArray();
            source.SetElementsAsFloat(values);
            Assert.True(CudaKernelOps.TryCopyHeadFirstToCache(cache, source, start, count, Capacity, circular: false));
            for (int h = 0; h < Heads; h++)
                for (int t = 0; t < count; t++)
                    for (int d = 0; d < Width; d++)
                    {
                        float x = values[(h * count + t) * Width + d];
                        byte[] bits = dtype == DType.Float32 ? BitConverter.GetBytes(x)
                            : BitConverter.GetBytes(BitConverter.HalfToUInt16Bits((System.Half)x));
                        bits.CopyTo(expected, ((h * Capacity + start + t) * Width + d) * dtype.Size());
                    }
        }
    }

    private static byte[] Seed(Tensor cache)
        => Enumerable.Range(0, checked((int)cache.Storage.ByteLength)).Select(i => (byte)(i * 73 + i / 7)).ToArray();

    private static byte[] Pack(byte[] source, long rowBytes, int start, int count)
    {
        int perHead = checked((int)(count * rowBytes));
        byte[] result = new byte[Heads * perHead];
        for (int h = 0; h < Heads; h++)
            Array.Copy(source, checked((int)((h * Capacity + start) * rowBytes)), result, h * perHead, perHead);
        return result;
    }

    private static unsafe void WriteBytes(Tensor cache, byte[] bytes)
    {
        fixed (byte* p = bytes) cache.Storage.CopyToStorage(0, (IntPtr)p, bytes.Length);
    }

    private static byte[] ReadBytes(Tensor cache)
    {
        byte[] bytes = new byte[checked((int)cache.Storage.ByteLength)];
        Marshal.Copy(cache.Storage.PtrAtElementForRead(0), bytes, 0, bytes.Length);
        return bytes;
    }

    private sealed class CheckoutAllocator : IAllocator
    {
        public BlasEnum BlasEnum => BlasEnum.DotNet;
        public int DeviceId => 0;
        public float GetAllocatedMemoryRatio() => 0;
        public Storage Allocate(DType dtype, long count) => new CheckoutStorage(this, dtype, count);
    }

    private sealed class CheckoutStorage(IAllocator allocator, DType dtype, long count) : CpuStorage(allocator, dtype, count)
    {
        internal int HostReadChecks, ReadOnlyCheckouts, WriteCheckouts, Uploads;
        internal bool HostDirty;
        public override void EnsureHostReadable() => HostReadChecks++;
        public override IntPtr PtrAtElement(long index)
        {
            WriteCheckouts++;
            HostDirty = true;
            return base.PtrAtElement(index);
        }
        public override IntPtr PtrAtElementForRead(long index)
        {
            ReadOnlyCheckouts++;
            return base.PtrAtElement(index);
        }
        public override void EnsureDeviceCurrent()
        {
            if (HostDirty) { Uploads++; HostDirty = false; }
        }
        internal void Seed(byte[] bytes) => Marshal.Copy(bytes, 0, buffer, bytes.Length);
        internal byte[] Bytes()
        {
            byte[] bytes = new byte[checked((int)ByteLength)];
            Marshal.Copy(buffer, bytes, 0, bytes.Length);
            return bytes;
        }
    }
}
