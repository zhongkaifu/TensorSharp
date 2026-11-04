// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

/// <summary>CPU payload and row-byte checks, not GPU upload or serving validation.</summary>
public sealed class NinferHostEmbeddingTests
{
    private sealed class Source : ICudaWeightSource, IDisposable
    {
        internal const int Prefix = 17;
        internal byte[] Bytes { get; }
        internal int Reads, LargestRead;
        internal bool Disposed, FailRead;
        internal Source(byte[] payload)
        {
            Bytes = new byte[Prefix + payload.Length + 31];
            Array.Fill(Bytes, (byte)0xa5); payload.CopyTo(Bytes, Prefix);
        }
        public void Read(long offset, IntPtr destination, long bytes)
        {
            ObjectDisposedException.ThrowIf(Disposed, this);
            if (FailRead) throw new IOException("Fixture read failure.");
            if (offset < 0 || bytes < 0 || offset + bytes > Bytes.Length) throw new ArgumentOutOfRangeException(nameof(offset));
            Reads++; LargestRead = Math.Max(LargestRead, checked((int)bytes));
            Marshal.Copy(Bytes, checked((int)offset), destination, checked((int)bytes));
        }
        public void Dispose() => Disposed = true;
    }

    private static byte[] Payload(int vocabulary, int width, Func<int, int, sbyte> code, Func<int, int, ushort> scale)
    {
        int codeBytes = checked(vocabulary * width), groups = width / 32;
        var bytes = new byte[checked(codeBytes + vocabulary * groups * 2)];
        for (int row = 0; row < vocabulary; row++)
        {
            for (int column = 0; column < width; column++) bytes[row * width + column] = unchecked((byte)code(row, column));
            for (int group = 0; group < groups; group++)
                BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(codeBytes + (row * groups + group) * 2, 2), scale(row, group));
        }
        return bytes;
    }

    private static CudaNinferProjectionDesc Descriptor(Source source, int vocabulary, int width) =>
        new(source, Source.Prefix, checked((long)vocabulary * width + (long)vocabulary * width / 16),
            vocabulary, width, CudaNinferProjectionFormat.W8G32, allowA4: false);

    // Independent rounding oracle: compares distances to adjacent representable
    // BF16 values in FP64, rather than using the production integer rounding bias.
    private static float NearestBfloat(float value)
    {
        ushort firstBits = (ushort)(BitConverter.SingleToUInt32Bits(value) >> 16);
        float first = BitConverter.UInt32BitsToSingle((uint)firstBits << 16);
        float second = BitConverter.UInt32BitsToSingle((uint)(firstBits + 1) << 16);
        double a = Math.Abs((double)value - first), b = Math.Abs((double)value - second);
        return a < b || (a == b && (firstBits & 1) == 0) ? first : second;
    }

    [Fact]
    public void EverySignedByteMatchesIndependentBfloatRoundingAcrossFiniteHalfScales()
    {
        ushort[] scales = { 0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3555, 0x3c00, 0xbc00, 0x3c04, 0x3c0c, 0xbc04, 0xbc0c, 0x7bff, 0xfbff };
        const int width = 256;
        byte[] bytes = Payload(scales.Length, width, (_, c) => (sbyte)(c - 128), (r, _) => scales[r]);
        using var source = new Source(bytes);
        using var host = new CudaNinferHostEmbedding(Descriptor(source, scales.Length, width));
        Assert.Equal(Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant(), host.PayloadSha256);
        int[] ids = Enumerable.Range(0, scales.Length).Reverse().ToArray();
        float[] actual = new float[ids.Length * width];
        host.GatherRows(ids, actual);
        for (int r = 0; r < ids.Length; r++)
            for (int c = 0; c < width; c++)
            {
                float exactProduct = (c - 128) * (float)BitConverter.UInt16BitsToHalf(scales[ids[r]]);
                Assert.Equal(BitConverter.SingleToInt32Bits(NearestBfloat(exactProduct)), BitConverter.SingleToInt32Bits(actual[r * width + c]));
            }
    }

    [Fact]
    public void EachGroupAndRepeatedNonsequentialRowUsesItsOriginalScaleWithoutSourceLifetimeDependency()
    {
        const int vocabulary = 7, width = 256;
        byte[] payload = Payload(vocabulary, width, (r, c) => (sbyte)((r * 17 + c * 7) % 256 - 128),
            (r, g) => BitConverter.HalfToUInt16Bits((Half)((r + 1) * (g + 1) / 32f)));
        using var source = new Source(payload);
        var host = new CudaNinferHostEmbedding(Descriptor(source, vocabulary, width));
        source.Dispose();
        int[] ids = { 6, 0, 3, 3, 1 };
        float[] output = Enumerable.Repeat(123.25f, ids.Length * width + 4).ToArray();
        host.GatherRows(ids, output.AsSpan(2));
        for (int r = 0; r < ids.Length; r++)
            for (int c = 0; c < width; c++)
            {
                float scale = (float)(Half)((ids[r] + 1) * (c / 32 + 1) / 32f);
                float exactProduct = ((ids[r] * 17 + c * 7) % 256 - 128) * scale;
                Assert.Equal(BitConverter.SingleToInt32Bits(NearestBfloat(exactProduct)), BitConverter.SingleToInt32Bits(output[2 + r * width + c]));
            }
        Assert.Equal(123.25f, output[0]); Assert.Equal(123.25f, output[1]);
        Assert.Equal(123.25f, output[^1]); Assert.Equal(123.25f, output[^2]);
        host.Dispose(); host.Dispose();
        Assert.Throws<ObjectDisposedException>(() => host.GatherRows(ids, output));
    }

    [Theory]
    [InlineData(0x7c00)] [InlineData(0xfc00)] [InlineData(0x7e00)] [InlineData(0x7fff)]
    public void NonfiniteHalfScalesAreRejectedWithoutTakingSourceOwnership(int bits)
    {
        byte[] payload = Payload(2, 128, (_, c) => (sbyte)c, (r, g) => r == 1 && g == 3 ? (ushort)bits : (ushort)0x3c00);
        using var source = new Source(payload);
        Assert.Throws<InvalidOperationException>(() => new CudaNinferHostEmbedding(Descriptor(source, 2, 128)));
        Assert.False(source.Disposed);
    }

    [Fact]
    public void InvalidRowsAndShortDestinationsLeaveTheDestinationUntouched()
    {
        using var source = new Source(Payload(2, 128, (_, _) => 1, (_, _) => 0x3c00));
        using var host = new CudaNinferHostEmbedding(Descriptor(source, 2, 128));
        var output = Enumerable.Repeat(13.25f, 256).ToArray();
        Assert.Throws<ArgumentOutOfRangeException>(() => host.GatherRows(new[] { 0, 2 }, output));
        Assert.Throws<ArgumentOutOfRangeException>(() => host.GatherRows(new[] { -1 }, output));
        Assert.Throws<ArgumentException>(() => host.GatherRows(new[] { 0, 1 }, output.AsSpan(1)));
        Assert.All(output, x => Assert.Equal(13.25f, x));
    }

    [Fact]
    public void ValidRepeatedGatherPerformsNoManagedAllocationOrSourceRead()
    {
        using var source = new Source(Payload(3, 128, (r, c) => (sbyte)(r + c - 64), (_, _) => 0x3c04));
        using var host = new CudaNinferHostEmbedding(Descriptor(source, 3, 128));
        int[] ids = { 2, 0, 1, 2 }; var output = new float[ids.Length * 128];
        for (int warmup = 0; warmup < 5; warmup++) host.GatherRows(ids, output);
        int reads = source.Reads;
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 64; i++) host.GatherRows(ids, output);
        long after = GC.GetAllocatedBytesForCurrentThread();
        Assert.Equal(before, after); Assert.Equal(reads, source.Reads);
    }

    [Fact]
    public void LoaderStreamsBoundedChunksIncludingAChunkThatCrossesTheCodeScaleBoundary()
    {
        const int vocabulary = 65537, width = 128;
        int codeBytes = vocabulary * width;
        byte[] payload = new byte[codeBytes + codeBytes / 16];
        payload.AsSpan(0, codeBytes).Fill(0x7f);
        for (int i = codeBytes; i < payload.Length; i += 2) BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(i, 2), 0x3c00);
        using var source = new Source(payload);
        using var host = new CudaNinferHostEmbedding(Descriptor(source, vocabulary, width));
        Assert.Equal(2, source.Reads);
        Assert.Equal(CudaNinferHostEmbedding.MaximumReadBytes, source.LargestRead);
        Assert.Equal(Convert.ToHexString(SHA256.HashData(payload)).ToLowerInvariant(), host.PayloadSha256);
        var output = new float[width]; host.GatherRows(new[] { vocabulary - 1 }, output);
        Assert.All(output, x => Assert.Equal(127f, x));
    }

    [Fact]
    public void ExplicitPayloadBudgetAndReadFailuresCannotRetainTheBorrowedSource()
    {
        using var source = new Source(Payload(1, 128, (_, _) => 1, (_, _) => 0x3c00));
        var descriptor = Descriptor(source, 1, 128);
        Assert.Throws<ArgumentException>(() => new CudaNinferHostEmbedding(descriptor, descriptor.Bytes - 1));
        Assert.Equal(0, source.Reads);
        source.FailRead = true;
        Assert.Throws<IOException>(() => new CudaNinferHostEmbedding(descriptor));
        Assert.False(source.Disposed);
        var tooLarge = new CudaNinferProjectionDesc(source, 0, 1_350_866_240, 248321, 5120, CudaNinferProjectionFormat.W8G32);
        Assert.Throws<ArgumentException>(() => new CudaNinferHostEmbedding(tooLarge));
    }

    [Theory]
    [InlineData(null, false)] [InlineData("0", false)] [InlineData("1", true)]
    public void HostPlacementIsAnExplicitOptin(string? raw, bool expected)
        => Assert.Equal(expected, Q35CudaEngine.HostNinferEmbeddingRequested(raw));

    [Theory]
    [InlineData("")] [InlineData(" ")] [InlineData("true")] [InlineData("2")]
    public void InvalidHostPlacementFlagCannotBeSilentlyIgnored(string raw)
        => Assert.Throws<ArgumentException>(() => Q35CudaEngine.HostNinferEmbeddingRequested(raw));

    [Fact]
    public void PinnedStagingCoversUbatchAndSmallUbatchDecodeWithinItsExplicitCap()
    {
        Assert.Equal(83_886_080, Q35CudaEngine.HostNinferEmbeddingStagingBytes(4096, 5120, out int rows));
        Assert.Equal(4096, rows);
        Assert.Equal(1_310_720, Q35CudaEngine.HostNinferEmbeddingStagingBytes(1, 5120, out rows));
        Assert.Equal(64, rows);
        Assert.Throws<NotSupportedException>(() => Q35CudaEngine.HostNinferEmbeddingStagingBytes(13108, 5120, out _));
    }
}
