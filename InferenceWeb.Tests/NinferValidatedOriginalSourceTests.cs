// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Buffers.Binary;
using System.Runtime.InteropServices;
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

/// <summary>Upload-validation state and exact byte contracts; no GPU load is exercised.</summary>
public sealed class NinferValidatedOriginalSourceTests
{
    private sealed class Original : ICudaNinferOriginalProjectionSource
    {
        internal byte[] Payload { get; }
        internal int Reads, EmptyReads, VirtualReads;
        internal bool Disposed, FailRead;
        internal long? BytesOverride;
        public long OriginalBytes => BytesOverride ?? Payload.Length;
        public int Rows { get; set; }
        public int Columns { get; set; }
        public CudaNinferOriginalRowMapping RowMapping { get; set; }
        internal Original(int rows = 128, int columns = 64)
        {
            Rows = rows; Columns = columns;
            int codes = checked(rows * columns / 2), scales = checked(rows * columns / 16);
            Payload = new byte[codes + scales + 4];
            for (int i = 0; i < codes; i++) Payload[i] = (byte)(i * 37 + 11);
            for (int i = 0; i < scales; i++) Payload[codes + i] = (byte)((i * 13) % 127);
            BinaryPrimitives.WriteUInt32LittleEndian(Payload.AsSpan(Payload.Length - 4), 0x40200000);
        }
        public void Read(long offset, IntPtr dst, long bytes) { VirtualReads++; throw new InvalidOperationException("Virtual planes must not be read by the original-object adapter."); }
        public void ReadOriginal(long offset, IntPtr dst, long bytes)
        {
            ObjectDisposedException.ThrowIf(Disposed, this);
            if (FailRead) throw new IOException("Original fixture read failed.");
            if (offset < 0 || bytes < 0 || offset > Payload.Length || bytes > Payload.Length - offset) throw new ArgumentOutOfRangeException(nameof(offset));
            Reads++; if (bytes == 0) EmptyReads++;
            if (bytes > 0) Marshal.Copy(Payload, checked((int)offset), dst, checked((int)bytes));
        }
    }

    private static CudaNinferProjectionDesc Descriptor(Original source, long sourceOffset = 0) => new(source, sourceOffset,
        checked((long)source.Rows * source.Columns / 2 + (long)source.Rows * source.Columns / 16),
        source.Rows, source.Columns, CudaNinferProjectionFormat.NvFp4Planes, weightScaleDivisor: 2.5f);

    private static byte[] Read(CudaNinferValidatedOriginalSource source, long offset, int bytes)
    {
        IntPtr allocation = Marshal.AllocHGlobal(bytes + 2);
        try
        {
            Marshal.WriteByte(allocation, 0x91); Marshal.WriteByte(allocation, bytes + 1, 0x92);
            source.Read(offset, allocation + 1, bytes);
            var result = new byte[bytes]; Marshal.Copy(allocation + 1, result, 0, bytes);
            Assert.Equal(0x91, Marshal.ReadByte(allocation)); Assert.Equal(0x92, Marshal.ReadByte(allocation, bytes + 1));
            return result;
        }
        finally { Marshal.FreeHGlobal(allocation); }
    }

    [Fact]
    public void CompleteUploadUsesRawBytesOnceAndDoesNotAlterTheObject()
    {
        var original = new Original(); var adapter = new CudaNinferValidatedOriginalSource(Descriptor(original));
        Assert.Equal(original.Payload, Read(adapter, 0, original.Payload.Length));
        Assert.Equal(1, original.Reads); Assert.Equal(0, original.VirtualReads); Assert.False(adapter.IsValidated);
        adapter.RequireCompleteValidation(); Assert.True(adapter.IsValidated);
        adapter.RequireCompleteValidation(); Assert.True(adapter.IsValidated);
        Assert.Equal(2, original.EmptyReads);
    }

    [Fact]
    public void ArbitraryChunksAndOrderRetainAllBytesAndCompleteCoverage()
    {
        var original = new Original(256, 128) { RowMapping = CudaNinferOriginalRowMapping.GateUpRows8 };
        var adapter = new CudaNinferValidatedOriginalSource(Descriptor(original));
        var pieces = new List<(int Start, int Bytes)>(); int[] sizes = { 1, 3, 127, 4095, 7, 65 };
        for (int start = 0, index = 0; start < original.Payload.Length; index++)
        {
            int count = Math.Min(sizes[index % sizes.Length], original.Payload.Length - start);
            pieces.Add((start, count)); start += count;
        }
        adapter.Read(0, IntPtr.Zero, 0); adapter.Read(adapter.Bytes, IntPtr.Zero, 0);
        foreach (var part in pieces.OrderByDescending(x => x.Start))
            Assert.Equal(original.Payload.AsSpan(part.Start, part.Bytes).ToArray(), Read(adapter, part.Start, part.Bytes));
        adapter.RequireCompleteValidation(); Assert.True(adapter.IsValidated); Assert.Equal(0, original.VirtualReads);
    }

    [Fact]
    public void DivisorCanStraddleFourSingleByteChunksInAnyOrder()
    {
        var original = new Original(); var adapter = new CudaNinferValidatedOriginalSource(Descriptor(original));
        _ = Read(adapter, 0, original.Payload.Length - 4);
        foreach (int index in new[] { 2, 0, 3, 1 })
            Assert.Equal(new[] { original.Payload[^4..][index] }, Read(adapter, adapter.Bytes - 4 + index, 1));
        adapter.RequireCompleteValidation(); Assert.True(adapter.IsValidated);
    }

    [Fact]
    public void MissingEvenOneScaleBytePermanentlyRefusesPublication()
    {
        var original = new Original(); var adapter = new CudaNinferValidatedOriginalSource(Descriptor(original));
        _ = Read(adapter, 0, original.Payload.Length - 5); _ = Read(adapter, adapter.Bytes - 4, 4);
        Assert.Throws<InvalidOperationException>(() => adapter.RequireCompleteValidation());
        Assert.False(adapter.IsValidated);
        int reads = original.Reads;
        Assert.Throws<InvalidOperationException>(() => Read(adapter, adapter.Bytes - 5, 1));
        Assert.Equal(reads, original.Reads);
    }

    [Theory]
    [InlineData(0x7f)] [InlineData(0x80)] [InlineData(0xff)]
    public void EveryForbiddenScaleClassFailsPermanentlyWithoutClipping(int invalid)
    {
        var original = new Original(); original.Payload[^5] = (byte)invalid;
        var adapter = new CudaNinferValidatedOriginalSource(Descriptor(original));
        Assert.Throws<InvalidDataException>(() => Read(adapter, 0, original.Payload.Length));
        Assert.Equal((byte)invalid, original.Payload[^5]); Assert.False(adapter.IsValidated);
        int reads = original.Reads;
        Assert.Throws<InvalidOperationException>(() => adapter.Read(0, IntPtr.Zero, 0));
        Assert.Throws<InvalidOperationException>(() => adapter.RequireCompleteValidation());
        Assert.Equal(reads, original.Reads);
    }

    [Theory]
    [InlineData(0u)] [InlineData(0x80000000u)] [InlineData(0xbf800000u)]
    [InlineData(0x7f800000u)] [InlineData(0x7fc01234u)] [InlineData(0x40400000u)]
    public void TrailerMustMatchTheExactFinitePositiveDescriptorBits(uint bits)
    {
        var original = new Original(); BinaryPrimitives.WriteUInt32LittleEndian(original.Payload.AsSpan(original.Payload.Length - 4), bits);
        var adapter = new CudaNinferValidatedOriginalSource(Descriptor(original));
        Assert.Throws<InvalidDataException>(() => Read(adapter, 0, original.Payload.Length));
        Assert.False(adapter.IsValidated); Assert.Throws<InvalidOperationException>(() => adapter.RequireCompleteValidation());
    }

    [Fact]
    public void RepeatedTrailerFragmentsMustRemainByteIdentical()
    {
        var original = new Original(); var adapter = new CudaNinferValidatedOriginalSource(Descriptor(original));
        _ = Read(adapter, adapter.Bytes - 4, 1); original.Payload[^4] = 1;
        Assert.Throws<InvalidDataException>(() => Read(adapter, adapter.Bytes - 4, 1));
        Assert.Throws<InvalidOperationException>(() => adapter.RequireCompleteValidation());
    }

    [Fact]
    public void ReadFailuresAndDisposedSourcesCannotBeRetriedIntoAPassingResult()
    {
        foreach (bool disposed in new[] { false, true })
        {
            var original = new Original { Disposed = disposed, FailRead = !disposed };
            var adapter = new CudaNinferValidatedOriginalSource(Descriptor(original));
            if (disposed) Assert.Throws<ObjectDisposedException>(() => adapter.Read(0, IntPtr.Zero, 0));
            else Assert.Throws<IOException>(() => adapter.Read(0, IntPtr.Zero, 0));
            original.Disposed = false; original.FailRead = false;
            Assert.Throws<InvalidOperationException>(() => Read(adapter, 0, original.Payload.Length));
            Assert.Throws<InvalidOperationException>(() => adapter.RequireCompleteValidation());
        }
    }

    [Fact]
    public void FragmentationCannotGrowValidationMemoryBeyondTheExplicitIntervalCap()
    {
        var original = new Original(256); var adapter = new CudaNinferValidatedOriginalSource(Descriptor(original));
        for (int i = 0; i < 4096; i++) _ = Read(adapter, 2L * i, 1);
        Assert.Throws<NotSupportedException>(() => Read(adapter, 8192, 1));
        Assert.Throws<InvalidOperationException>(() => adapter.RequireCompleteValidation());
    }

    [Fact]
    public void GeometryOffsetMappingAndStagingBoundsAreValidatedBeforeReading()
    {
        var offset = new Original(); Assert.Throws<NotSupportedException>(() => new CudaNinferValidatedOriginalSource(Descriptor(offset, 1)));
        var length = new Original { BytesOverride = 99 }; Assert.Throws<NotSupportedException>(() => new CudaNinferValidatedOriginalSource(Descriptor(length)));
        var mapping = new Original { RowMapping = CudaNinferOriginalRowMapping.GateUpRows8 };
        Assert.Throws<ArgumentException>(() => new CudaNinferValidatedOriginalSource(Descriptor(mapping)));
        var unknown = new Original { RowMapping = (CudaNinferOriginalRowMapping)99 };
        Assert.Throws<ArgumentException>(() => new CudaNinferValidatedOriginalSource(Descriptor(unknown)));
        var huge = new Original { Rows = 4096, Columns = 65536, BytesOverride = 150_994_948 };
        Assert.Throws<NotSupportedException>(() => new CudaNinferValidatedOriginalSource(Descriptor(huge)));
        Assert.All(new[] { offset, length, mapping, unknown, huge }, source => Assert.Equal(0, source.Reads));
    }

    [Fact]
    public void ReadBoundsFailuresAlsoPermanentlyFaultTheUpload()
    {
        var original = new Original(); var adapter = new CudaNinferValidatedOriginalSource(Descriptor(original));
        Assert.Throws<ArgumentOutOfRangeException>(() => adapter.Read(adapter.Bytes, IntPtr.Zero, 1));
        Assert.Throws<InvalidOperationException>(() => adapter.RequireCompleteValidation()); Assert.Equal(0, original.Reads);
    }
}
