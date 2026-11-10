// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>
/// The draft vocabulary subset (TS_DFLASH_DRAFT_VOCAB): Strata's draft_vocab*.bin format, the per-token row
/// gather the fused DFlash draft scores, and the map back to token ids.
/// </summary>
public class DraftVocabSubsetTests
{
    private static byte[] Bytes(params int[] ids) => MemoryMarshal.AsBytes(ids.AsSpan()).ToArray();

    [Fact]
    public void Parse_AcceptsAscendingIdsInRange()
    {
        int[] map = DraftVocabSubset.Parse(Bytes(0, 3, 7, 9), vocab: 10, minTokens: 2, out string problem);
        Assert.Null(problem);
        Assert.Equal(new[] { 0, 3, 7, 9 }, map);
    }

    [Fact]
    public void Parse_AcceptsTwoSortedRuns()
    {
        // Strata's default draft_vocab.bin: its English set, then a second sorted run of the other tokens.
        int[] map = DraftVocabSubset.Parse(Bytes(1, 4, 8, 0, 2, 9), vocab: 10, minTokens: 2, out string problem);
        Assert.Null(problem);
        Assert.Equal(new[] { 1, 4, 8, 0, 2, 9 }, map);
    }

    [Theory]
    [InlineData(new[] { 0, 3, 3 }, "listed twice")]
    [InlineData(new[] { 5, 2, 5 }, "listed twice")]
    [InlineData(new[] { 0, 10 }, "out of range")]
    [InlineData(new[] { -1, 2 }, "out of range")]
    [InlineData(new[] { 4 }, "fewer than")]
    public void Parse_RejectsWhatCannotServeAsADraftVocabulary(int[] ids, string reason)
    {
        Assert.Null(DraftVocabSubset.Parse(Bytes(ids), vocab: 10, minTokens: 2, out string problem));
        Assert.Contains(reason, problem);
    }

    [Fact]
    public void Parse_RejectsAPartialId()
    {
        Assert.Null(DraftVocabSubset.Parse(new byte[] { 1, 0, 0, 0, 2, 0 }, vocab: 10, minTokens: 1, out string problem));
        Assert.Contains("int32", problem);
    }

    [Fact]
    public void Load_ReportsAMissingFile()
    {
        string path = Path.Combine(Path.GetTempPath(), "ts-draft-vocab-missing-" + Guid.NewGuid().ToString("N") + ".bin");
        Assert.Null(DraftVocabSubset.Load(path, 10, 1, out string problem));
        Assert.False(string.IsNullOrEmpty(problem));
    }

    [Fact]
    public unsafe void Gather_CopiesEachSelectedRowInMapOrder()
    {
        const int rows = 9, rowBytes = 37;   // an odd row size: quantized rows are whole blocks, not aligned words
        byte[] table = new byte[rows * rowBytes];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < rowBytes; c++)
                table[r * rowBytes + c] = (byte)(r * 31 + c);
        int[] map = { 1, 4, 8 };
        fixed (byte* src = table)
        {
            IntPtr sub = DraftVocabSubset.Gather((IntPtr)src, rowBytes, map, out long bytes);
            try
            {
                Assert.Equal((long)map.Length * rowBytes, bytes);
                Assert.Equal(0, (long)sub % 4096);   // the native tensor cache wraps it on unified memory
                var got = new ReadOnlySpan<byte>((void*)sub, (int)bytes);
                for (int i = 0; i < map.Length; i++)
                    Assert.True(got.Slice(i * rowBytes, rowBytes).SequenceEqual(table.AsSpan(map[i] * rowBytes, rowBytes)));
            }
            finally { DraftVocabSubset.Free(sub, bytes); }
        }
    }

    [Fact]
    public void MapToTokens_MapsSubsetIndicesAndLeavesTheRest()
    {
        int[] map = { 2, 5, 11, 40 };
        int[] ids = { 0, 3, 1, -1, 9, 2 };
        DraftVocabSubset.MapToTokens(map, ids, 5);
        Assert.Equal(new[] { 2, 40, 5, -1, 9, 2 }, ids);   // -1 and 9 are not subset indices; the 6th is past count
    }

    [Fact]
    public void MapToTokens_WithoutASubsetChangesNothing()
    {
        int[] ids = { 7, 8 };
        DraftVocabSubset.MapToTokens(null, ids, 2);
        Assert.Equal(new[] { 7, 8 }, ids);
    }
}
