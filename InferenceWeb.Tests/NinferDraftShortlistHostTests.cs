// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Buffers.Binary;
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

/// <summary>Original byte/layout/admission policies only; no CUDA or model quality claim.</summary>
public sealed class NinferDraftShortlistHostTests
{
    private static void Codes(Q35CudaEngine.NinferShortlistPayloadValidator validator, int trailing = 0)
    {
        byte[] bytes = Enumerable.Repeat((byte)0xff, 1 << 20).ToArray();
        while (validator.BytesRead < 335544320L - trailing)
        {
            int take = (int)Math.Min(bytes.Length, 335544320L - trailing - validator.BytesRead);
            validator.Accept(validator.BytesRead, bytes.AsSpan(0, take));
        }
    }
    private static void Scales(Q35CudaEngine.NinferShortlistPayloadValidator validator, int stride = 65536)
    {
        byte[] bytes = new byte[stride];
        while (validator.BytesRead < 356515840L)
        {
            int take = (int)Math.Min(bytes.Length, 356515840L - validator.BytesRead);
            validator.Accept(validator.BytesRead, bytes.AsSpan(0, take));
        }
    }

    [Fact]
    public void OriginalGeometryIsNotNvfp4OrGgmlQ4K()
    {
        Assert.Equal(131072, Q35CudaEngine.NinferShortlistRows);
        Assert.Equal(5120, Q35CudaEngine.NinferShortlistColumns);
        Assert.Equal(335544320L, Q35CudaEngine.NinferShortlistCodeBytes);
        Assert.Equal(20971520L, Q35CudaEngine.NinferShortlistScaleBytes);
        Assert.Equal(356515840L, Q35CudaEngine.NinferShortlistWeightBytes);
        Assert.Equal(134217728, Q35CudaEngine.NinferShortlistLoadChunkBytes);
        Assert.Equal(80, Q35CudaEngine.NinferShortlistColumns / 64);
    }

    [Theory]
    [InlineData(0x0000)]
    [InlineData(0x8000)]
    [InlineData(0x0001)]
    [InlineData(0x03ff)]
    [InlineData(0x7bff)]
    [InlineData(0xfbff)]
    [InlineData(0x3c00)]
    [InlineData(0xbc00)]
    public void FiniteOriginalScaleBitsArePreservedAcrossSplitBytes(int bits)
    {
        var validator = new Q35CudaEngine.NinferShortlistPayloadValidator(); Codes(validator, trailing: 1);
        validator.Accept(335544319, new byte[] { 0xff, (byte)bits }); // Last code plus low scale byte.
        validator.Accept(335544321, ReadOnlySpan<byte>.Empty);
        Assert.Equal(0, validator.ValidatedScales);
        validator.Accept(335544321, new[] { (byte)(bits >> 8) });
        Assert.Equal(1, validator.ValidatedScales);
        Assert.Equal(335544322, validator.BytesRead);
    }

    [Theory]
    [InlineData(0x7c00)]
    [InlineData(0xfc00)]
    [InlineData(0x7c01)]
    [InlineData(0x7fff)]
    [InlineData(0xffff)]
    public void AllNonfiniteScaleClassesFailPermanentlyEvenWhenSplit(int bits)
    {
        var validator = new Q35CudaEngine.NinferShortlistPayloadValidator(); Codes(validator);
        validator.Accept(335544320, new[] { (byte)bits });
        Assert.Throws<InvalidDataException>(() => validator.Accept(335544321, new[] { (byte)(bits >> 8) }));
        Assert.Throws<InvalidOperationException>(() => validator.Accept(335544321, new byte[] { 0 }));
        Assert.Throws<InvalidOperationException>(validator.RequireComplete);
    }

    [Theory]
    [InlineData(257)]
    [InlineData(65536)]
    public void CompleteCoverageValidatesEveryScaleWithOddOrEvenChunks(int chunk)
    {
        var validator = new Q35CudaEngine.NinferShortlistPayloadValidator(); Codes(validator); Scales(validator, chunk);
        validator.RequireComplete(); validator.Accept(356515840, ReadOnlySpan<byte>.Empty); validator.RequireComplete();
        Assert.Equal(356515840, validator.BytesRead);
        Assert.Equal(10485760, validator.ValidatedScales);
    }

    [Fact]
    public void MissingFinalScaleByteAndEarlyPublishArePermanentFailures()
    {
        var validator = new Q35CudaEngine.NinferShortlistPayloadValidator(); Codes(validator);
        byte[] bytes = new byte[65536];
        while (validator.BytesRead < 356515839L)
            validator.Accept(validator.BytesRead, bytes.AsSpan(0, (int)Math.Min(bytes.Length, 356515839L - validator.BytesRead)));
        Assert.Equal(10485759, validator.ValidatedScales);
        Assert.Throws<InvalidOperationException>(validator.RequireComplete);
        Assert.Throws<InvalidOperationException>(() => validator.Accept(356515839, new byte[] { 0 }));
        var empty = new Q35CudaEngine.NinferShortlistPayloadValidator();
        Assert.Throws<InvalidOperationException>(empty.RequireComplete);
        Assert.Throws<InvalidOperationException>(() => empty.Accept(0, new byte[] { 0 }));
    }

    [Fact]
    public void LastScaleIsValidatedAndReadOverlapCannotBeReplayed()
    {
        var validator = new Q35CudaEngine.NinferShortlistPayloadValidator(); Codes(validator);
        byte[] bytes = new byte[65536];
        while (validator.BytesRead < 356515838L)
            validator.Accept(validator.BytesRead, bytes.AsSpan(0, (int)Math.Min(bytes.Length, 356515838L - validator.BytesRead)));
        Assert.Equal(10485759, validator.ValidatedScales);
        Assert.Throws<InvalidDataException>(() => validator.Accept(356515838, new byte[] { 0, 0x7c }));
        Assert.Throws<InvalidOperationException>(validator.RequireComplete);
        var overlap = new Q35CudaEngine.NinferShortlistPayloadValidator(); overlap.Accept(0, new byte[6]);
        Assert.Throws<ArgumentOutOfRangeException>(() => overlap.Accept(4, new byte[2]));
        Assert.Throws<InvalidOperationException>(() => overlap.Accept(6, new byte[2]));
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(1)]
    [InlineData(356515840)]
    public void MissingOverlappingOrForeignReadOffsetFailsClosed(long offset)
    {
        var validator = new Q35CudaEngine.NinferShortlistPayloadValidator();
        Assert.Throws<ArgumentOutOfRangeException>(() => validator.Accept(offset, new byte[] { 0 }));
        Assert.Throws<InvalidOperationException>(() => validator.Accept(0, new byte[] { 0 }));
    }

    [Fact]
    public void OriginalMapKeepsDuplicateRowsAndExactPhysicalIdEndpoints()
    {
        int[] ids = new int[131072]; ids[0] = 248319; ids[1] = 248319; ids[65535] = 248077; ids[^1] = 17;
        byte[] bytes = Q35CudaEngine.EncodeNinferShortlistMap(ids);
        Assert.Equal(524288, bytes.Length);
        Assert.Equal(248319, BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(0, 4)));
        Assert.Equal(248319, BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(4, 4)));
        Assert.Equal(248077, BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(65535 * 4, 4)));
        Assert.Equal(17, BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(bytes.Length - 4, 4)));
        ids[0] = 0; Assert.Equal(248319, BinaryPrimitives.ReadInt32LittleEndian(bytes));
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(248320)]
    [InlineData(int.MaxValue)]
    public void OriginalMapRefusesIdsOutsidePhysicalTargetVocabulary(int id)
    {
        int[] ids = new int[131072]; ids[^1] = id;
        Assert.Throws<InvalidDataException>(() => Q35CudaEngine.EncodeNinferShortlistMap(ids));
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.EncodeNinferShortlistMap(new int[131071]));
    }

    [Fact]
    public void StateNamespaceBindsPolicyAndEveryIndependentLoadedIdentity()
    {
        string a = new('a', 64), b = new('b', 64), c = new('c', 64), d = new('d', 64);
        string identity = Q35CudaEngine.NinferShortlistStateSuffix(a, b, c);
        Assert.Contains("original-q4g64-f16s-row-split-k128-e600-v1", identity);
        Assert.Equal(identity, Q35CudaEngine.NinferShortlistStateSuffix(a.ToUpperInvariant(), b, c));
        Assert.NotEqual(identity, Q35CudaEngine.NinferShortlistStateSuffix(d, b, c));
        Assert.NotEqual(identity, Q35CudaEngine.NinferShortlistStateSuffix(a, d, c));
        Assert.NotEqual(identity, Q35CudaEngine.NinferShortlistStateSuffix(a, b, d));
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.NinferShortlistStateSuffix(a[..63], b, c));
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.NinferShortlistStateSuffix(a, new string('z', 64), c));
    }
}
