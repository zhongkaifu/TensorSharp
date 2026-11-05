// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>Real bounded-reader byte tests. No CUDA repack or GPU inference is exercised.</summary>
public sealed class NinferOriginalProjectionSourceTests
{
    private sealed class Fixture : IDisposable
    {
        private readonly NinferV3File _file;
        internal NinferArtifactReader Reader { get; }
        internal NinferTensorDescriptor Tensor => Reader.GetBinding("projection").Object;
        internal byte[] Original { get; }
        internal Fixture(int rows = 128, int columns = 64)
        {
            int codes = checked(rows * columns / 2), scales = checked(rows * columns / 16);
            Original = new byte[codes + scales + 4];
            for (int i = 0; i < codes; i++) Original[i] = unchecked((byte)(i * 29 + (i >> 3) * 11));
            for (int i = 0; i < scales; i++) Original[codes + i] = (byte)((i * 13 + 7) % 127);
            BinaryPrimitives.WriteUInt32LittleEndian(Original.AsSpan(codes + scales), 0x40200000);
            var b = new NinferV3Builder();
            b.Bind("projection", b.TensorBytes("nvfp4", new[] { (ulong)rows, (ulong)columns }, Original));
            b.Resource("text", "unrelated", Enumerable.Repeat((byte)0xee, 32).ToArray());
            _file = b.Write();
            Reader = new NinferArtifactReader(_file.Path);
        }
        internal NinferNvfp4WeightSource Source(NinferNvfp4RowMapping mapping = NinferNvfp4RowMapping.Identity)
            => new(new NinferNvfp4PlaneReader(Reader, Tensor, mapping: mapping));
        public void Dispose() { Reader.Dispose(); _file.Dispose(); }
    }

    private static byte[] Read(ICudaNinferOriginalProjectionSource source, long offset, int bytes)
    {
        IntPtr allocation = Marshal.AllocHGlobal(bytes + 2);
        try
        {
            Marshal.WriteByte(allocation, 0x91); Marshal.WriteByte(allocation, bytes + 1, 0x92);
            source.ReadOriginal(offset, allocation + 1, bytes);
            var result = new byte[bytes]; Marshal.Copy(allocation + 1, result, 0, bytes);
            Assert.Equal(0x91, Marshal.ReadByte(allocation)); Assert.Equal(0x92, Marshal.ReadByte(allocation, bytes + 1));
            return result;
        }
        finally { Marshal.FreeHGlobal(allocation); }
    }

    [Theory]
    [InlineData(128, NinferNvfp4RowMapping.Identity, CudaNinferOriginalRowMapping.Identity)]
    [InlineData(256, NinferNvfp4RowMapping.GateUpRows8, CudaNinferOriginalRowMapping.GateUpRows8)]
    [InlineData(14336, NinferNvfp4RowMapping.AttentionQueryGateKeyValue, CudaNinferOriginalRowMapping.AttentionQueryGateKeyValue)]
    public void CapabilityPreservesOriginalObjectAndDeclaresMappingWithoutApplyingIt(int rows, NinferNvfp4RowMapping mapping,
        CudaNinferOriginalRowMapping expected)
    {
        using var fixture = new Fixture(rows);
        using var source = fixture.Source(mapping);
        ICudaNinferOriginalProjectionSource raw = source;
        Assert.Equal(rows, raw.Rows); Assert.Equal(64, raw.Columns); Assert.Equal(expected, raw.RowMapping);
        Assert.Equal(fixture.Original.Length, raw.OriginalBytes);
        Assert.Equal(source.Bytes + 4, raw.OriginalBytes);
        long reads = fixture.Reader.ReadOperations;
        Assert.Equal(fixture.Original, Read(raw, 0, fixture.Original.Length));
        Assert.Equal(1, fixture.Reader.ReadOperations - reads); // No 128-row tile traversal or conversion.
        Assert.Equal(fixture.Original.AsSpan(4093, 19).ToArray(), Read(raw, 4093, 19));
        Assert.Equal(new byte[] { 0, 0, 0x20, 0x40 }, Read(raw, raw.OriginalBytes - 4, 4));
    }

    [Fact]
    public void OriginalReadCannotReachAnotherObjectOrAcceptAnInvalidDestination()
    {
        using var fixture = new Fixture(); using var source = fixture.Source();
        ICudaNinferOriginalProjectionSource raw = source;
        Assert.Throws<ArgumentOutOfRangeException>(() => Read(raw, -1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => Read(raw, raw.OriginalBytes - 1, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => Read(raw, raw.OriginalBytes + 1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => raw.ReadOriginal(0, IntPtr.Zero, -1));
        Assert.Throws<ArgumentException>(() => raw.ReadOriginal(0, IntPtr.Zero, 1));
        Assert.Throws<ArgumentException>(() => raw.ReadOriginal(0, new IntPtr(long.MaxValue), 2));
        raw.ReadOriginal(raw.OriginalBytes, IntPtr.Zero, 0);
    }

    [Fact]
    public void ForeignReadersNeverManufactureTheOriginalByteCapability()
    {
        using var first = new Fixture(); using var second = new Fixture();
        Assert.Throws<ArgumentException>(() => new NinferNvfp4WeightSource(new NinferNvfp4PlaneReader(first.Reader, second.Tensor)));
    }

    [Fact]
    public void OriginalReadChecksSourcePlaneAndUnderlyingReaderLifetimesIncludingEmptyReads()
    {
        using var first = new Fixture();
        var disposedSource = first.Source(); disposedSource.Dispose();
        Assert.Throws<ObjectDisposedException>(() => ((ICudaNinferOriginalProjectionSource)disposedSource).ReadOriginal(0, IntPtr.Zero, 0));
        var plane = new NinferNvfp4PlaneReader(first.Reader, first.Tensor);
        using var sourceWithDisposedPlane = new NinferNvfp4WeightSource(plane); plane.Dispose();
        Assert.Throws<ObjectDisposedException>(() => ((ICudaNinferOriginalProjectionSource)sourceWithDisposedPlane).ReadOriginal(0, IntPtr.Zero, 0));

        using var second = new Fixture(); using var sourceWithDisposedReader = second.Source(); second.Reader.Dispose();
        Assert.Throws<ObjectDisposedException>(() => ((ICudaNinferOriginalProjectionSource)sourceWithDisposedReader).ReadOriginal(0, IntPtr.Zero, 0));
    }

    [Fact]
    public void OriginalAndConvertedReadsCanInterleaveWithoutChangingEitherByteContract()
    {
        using var fixture = new Fixture(256); using var source = fixture.Source(NinferNvfp4RowMapping.GateUpRows8);
        using var reference = new NinferNvfp4PlaneReader(fixture.Reader, fixture.Tensor, mapping: NinferNvfp4RowMapping.GateUpRows8);
        var tile = reference.ReadTile(0);
        byte[] expected = tile.Codes.ToArray();
        IntPtr buffer = Marshal.AllocHGlobal(expected.Length);
        try
        {
            source.Read(0, buffer, expected.Length);
            var before = new byte[expected.Length]; Marshal.Copy(buffer, before, 0, before.Length); Assert.Equal(expected, before);
            Assert.Equal(fixture.Original, Read(source, 0, fixture.Original.Length));
            source.Read(0, buffer, expected.Length);
            var after = new byte[expected.Length]; Marshal.Copy(buffer, after, 0, after.Length); Assert.Equal(expected, after);
        }
        finally { Marshal.FreeHGlobal(buffer); }
    }
}
