// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Buffers.Binary;
using System.Runtime.InteropServices;
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

/// <summary>Original anchor/mask CPU gathering only; no new upload/device qualification.</summary>
public sealed class NinferDraftHostEmbeddingTests
{
    private const int Width = 128, Vocabulary = 7, Mask = 5, Prefix = 13;
    private static readonly ushort[] ScaleBits = { 0x8000, 0x3c04, 0xbc0c, 0x0001, 0x7bff };
    private sealed class Source : ICudaWeightSource
    {
        internal byte[] Bytes { get; }
        internal int Reads;
        internal Source()
        {
            int codeBytes = Width * Vocabulary;
            Bytes = new byte[Prefix + codeBytes + codeBytes / 16];
            for (int row = 0; row < Vocabulary; row++)
            {
                for (int column = 0; column < Width; column++)
                    Bytes[Prefix + row * Width + column] = unchecked((byte)Code(row, column));
                for (int group = 0; group < Width / 32; group++)
                    BinaryPrimitives.WriteUInt16LittleEndian(Bytes.AsSpan(Prefix + codeBytes + (row * (Width / 32) + group) * 2, 2), ScaleBits[(row + group) % ScaleBits.Length]);
            }
        }
        public void Read(long offset, IntPtr destination, long bytes)
        {
            Reads++; Marshal.Copy(Bytes, checked((int)offset), destination, checked((int)bytes));
        }
    }
    private static sbyte Code(int row, int column) => (sbyte)((row * 37 + column) % 256 - 128);
    private static CudaNinferHostEmbedding Host(Source source) => new(new CudaNinferProjectionDesc(source,
        Prefix, (long)Width * Vocabulary + (long)Width * Vocabulary / 16, Vocabulary, Width, CudaNinferProjectionFormat.W8G32, allowA4: false));
    private static int ExpectedBits(int row, int column)
    {
        float value = Code(row, column) * (float)BitConverter.UInt16BitsToHalf(ScaleBits[(row + column / 32) % ScaleBits.Length]);
        ushort lowerBits = (ushort)(BitConverter.SingleToUInt32Bits(value) >> 16);
        float lower = BitConverter.UInt32BitsToSingle((uint)lowerBits << 16), upper = BitConverter.UInt32BitsToSingle((uint)(lowerBits + 1) << 16);
        double a = Math.Abs((double)value - lower), b = Math.Abs((double)value - upper);
        return BitConverter.SingleToInt32Bits(a < b || a == b && (lowerBits & 1) == 0 ? lower : upper);
    }

    [Theory]
    [InlineData(1, 2)] [InlineData(1, 8)] [InlineData(2, 4)] [InlineData(3, 8)] [InlineData(4, 8)] [InlineData(8, 8)]
    public void ActualGatherUsesAnchorThenMaskForEveryPhysicalColumn(int lanes, int block)
    {
        var source = new Source(); using var host = Host(source);
        int[] anchors = Enumerable.Range(0, lanes).Select(q => (6 - q + Vocabulary) % Vocabulary).ToArray();
        int rows = lanes * block; float[] values = Enumerable.Repeat(13.25f, rows * Width + 6).ToArray();
        Assert.Equal(rows, Q35CudaEngine.GatherHostNinferDraftEmbedding(host, anchors, block, Mask, values.AsSpan(3, rows * Width)));
        for (int row = 0; row < rows; row++)
            for (int column = 0; column < Width; column++)
                Assert.Equal(ExpectedBits(row % block == 0 ? anchors[row / block] : Mask, column), BitConverter.SingleToInt32Bits(values[3 + row * Width + column]));
        Assert.All(values.Take(3).Concat(values.TakeLast(3)), value => Assert.Equal(13.25f, value));
    }

    [Fact]
    public void InvalidLastAnchorMaskCapacityAndShapeCannotWriteAnyGatheredRow()
    {
        var source = new Source(); using var host = Host(source); float[] values = Enumerable.Repeat(13.25f, 64 * Width).ToArray();
        foreach (int token in new[] { -1, Vocabulary, int.MaxValue })
            Assert.Throws<ArgumentOutOfRangeException>(() => Q35CudaEngine.GatherHostNinferDraftEmbedding(host, new[] { 2, token }, 8, Mask, values));
        Assert.Throws<ArgumentOutOfRangeException>(() => Q35CudaEngine.GatherHostNinferDraftEmbedding(host, new[] { 2 }, 8, Vocabulary, values));
        Assert.Throws<ArgumentOutOfRangeException>(() => Q35CudaEngine.GatherHostNinferDraftEmbedding(host, Array.Empty<int>(), 8, Mask, values));
        Assert.Throws<ArgumentOutOfRangeException>(() => Q35CudaEngine.GatherHostNinferDraftEmbedding(host, new int[9], 8, Mask, values));
        foreach (int block in new[] { 0, 1, 9, int.MaxValue })
            Assert.Throws<ArgumentOutOfRangeException>(() => Q35CudaEngine.GatherHostNinferDraftEmbedding(host, new[] { 2 }, block, Mask, values));
        Assert.Throws<ArgumentOutOfRangeException>(() => Q35CudaEngine.GatherHostNinferDraftEmbedding(host, new[] { 2, 4 }, 8, Mask, values.AsSpan(0, 16 * Width - 1)));
        Assert.All(values, value => Assert.Equal(13.25f, value));
    }

    [Fact]
    public void ValidGatherAfterWarmupAllocatesNothingAndNeverReadsBorrowedSourceAgain()
    {
        var source = new Source(); using var host = Host(source); int[] anchors = { 6, 0, 3, 3 }; float[] values = new float[32 * Width];
        for (int i = 0; i < 5; i++) Q35CudaEngine.GatherHostNinferDraftEmbedding(host, anchors, 8, Mask, values);
        int reads = source.Reads; long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 64; i++) Q35CudaEngine.GatherHostNinferDraftEmbedding(host, anchors, 8, Mask, values);
        Assert.Equal(before, GC.GetAllocatedBytesForCurrentThread()); Assert.Equal(reads, source.Reads);
    }

    [Fact]
    public void DisposedPayloadCannotWriteCallerDestination()
    {
        var source = new Source(); var host = Host(source); host.Dispose(); float[] values = Enumerable.Repeat(13.25f, 8 * Width).ToArray();
        Assert.Throws<ObjectDisposedException>(() => Q35CudaEngine.GatherHostNinferDraftEmbedding(host, new[] { 3 }, 8, Mask, values));
        Assert.All(values, value => Assert.Equal(13.25f, value));
    }

    [Fact]
    public void OriginalFullSizeProposalFitsThePreexisting128RowStaging()
    {
        Assert.Equal(64, Q35CudaEngine.HostNinferDraftEmbeddingRows(new int[8], 8, 248070, 248320, 128));
        Assert.Equal(2621440, Q35CudaEngine.HostNinferEmbeddingStagingBytes(128, 5120, out int capacity));
        Assert.Equal(128, capacity);
        Assert.Throws<ArgumentOutOfRangeException>(() => Q35CudaEngine.HostNinferDraftEmbeddingRows(new int[8], 8, 248070, 248320, 63));
    }
}
