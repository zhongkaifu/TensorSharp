// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Regression tests for the KV block snapshot helper.
//
// The bug these pin down: Muse-Glimmer sizes its 39 sliding-window layers as a
// RING of (window + prefill chunk + 1) rows while its 13 full layers get the whole
// context. KvBlockTransfer addressed every layer as if row index == absolute token
// position, so the first block whose start passed the ring size read
// (numKVHeads - 1) * ringRows + startToken rows into a buffer that only has
// numKVHeads * ringRows - i.e. straight past the end of the allocation. The server
// died with
//
//   System.AccessViolationException: Attempted to read or write protected memory.
//      at System.SpanHelpers.Memmove(...)
//      at TensorSharp.Models.KvBlockTransfer.CopyOneOut(...)
//      at TensorSharp.Models.MuseGlimmerModel.TryExtractKVBlock(...)
//      at TensorSharp.Runtime.Scheduling.BatchExecutor.CaptureNewlyFullBlocks(...)
//
// once a single sequence reached ringRows + blockSize tokens.
//
// These run on the CPU allocator with no GGUF, so they are unconditional.
using System;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public class KvBlockTransferRingTests : IDisposable
{
    private const int NumKVHeads = 3;
    private const int HeadDim = 4;

    private readonly IAllocator _alloc = new CpuAllocator(BlasEnum.DotNet);
    private readonly System.Collections.Generic.List<Tensor> _owned = new();

    public void Dispose()
    {
        foreach (var t in _owned) t.Dispose();
        _owned.Clear();
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private Tensor NewCache(int rows)
    {
        var t = new Tensor(_alloc, DType.Float32, NumKVHeads, rows, HeadDim);
        Ops.Fill(t, 0f);
        _owned.Add(t);
        return t;
    }

    /// <summary>A value unique to (tensor, head, row, dim) so a misplaced copy is
    /// impossible to mistake for a correct one.</summary>
    private static float Cell(int tag, int head, int row, int dim)
        => tag * 1_000_000f + head * 10_000f + row * 10f + dim;

    /// <summary>Write the row that absolute position <paramref name="pos"/> occupies,
    /// exactly as a forward pass would: linear layers store it at row pos, ring layers
    /// at row pos % rows.</summary>
    private static void WritePosition(Tensor t, int tag, int pos, int ringRows)
    {
        int rows = (int)t.Sizes[1];
        int row = ringRows > 0 && rows == ringRows ? pos % rows : pos;
        for (int h = 0; h < NumKVHeads; h++)
            for (int d = 0; d < HeadDim; d++)
                t.SetElementAsFloat(Cell(tag, h, pos, d), h, row, d);
    }

    private static void WriteSequence(Tensor[] k, Tensor[] v, int seqLen, int ringRows)
    {
        for (int pos = 0; pos < seqLen; pos++)
        {
            for (int l = 0; l < k.Length; l++)
            {
                WritePosition(k[l], 2 * l, pos, ringRows);
                WritePosition(v[l], 2 * l + 1, pos, ringRows);
            }
        }
    }

    private static byte[] Snapshot(Tensor[] k, Tensor[] v)
    {
        var bytes = new System.Collections.Generic.List<byte>();
        foreach (var t in Concat(k, v))
        {
            int rows = (int)t.Sizes[1];
            for (int h = 0; h < NumKVHeads; h++)
                for (int r = 0; r < rows; r++)
                    for (int d = 0; d < HeadDim; d++)
                        bytes.AddRange(BitConverter.GetBytes(t.GetElementAsFloat(h, r, d)));
        }
        return bytes.ToArray();
    }

    private static System.Collections.Generic.IEnumerable<Tensor> Concat(Tensor[] k, Tensor[] v)
    {
        for (int l = 0; l < k.Length; l++) { yield return k[l]; yield return v[l]; }
    }

    // ------------------------------------------------------------------
    // The crash
    // ------------------------------------------------------------------

    /// <summary>
    /// The exact shape that faulted: a 512-row ring layer next to a 2048-row full
    /// layer, asked for the block starting at 512. Addressed linearly this walks off
    /// the end of the ring tensor. It must be refused, not attempted.
    /// </summary>
    [Fact]
    public void Extract_BlockPastAShortLayersRows_IsRefusedWhenAddressedLinearly()
    {
        var k = new[] { NewCache(512), NewCache(2048) };
        var v = new[] { NewCache(512), NewCache(2048) };
        WriteSequence(k, v, 768, ringRows: 512);

        long bytes = KvBlockTransfer.ComputeBlockByteSize(k, v, 256);
        var dst = new byte[bytes];

        // ringRows: 0 => every layer is linear, and layer 0 simply cannot address
        // position 512. Refusal is the only safe answer.
        Assert.False(KvBlockTransfer.Extract(_alloc, k, v, 768, 512, 256, dst, ringRows: 0));
    }

    /// <summary>Same block, now with the ring declared: it is addressable, and the
    /// wrapped rows come back with the values the forward pass wrote there.</summary>
    [Fact]
    public void Extract_BlockPastTheRing_ReadsTheWrappedRows()
    {
        const int ring = 512;
        var k = new[] { NewCache(ring), NewCache(2048) };
        var v = new[] { NewCache(ring), NewCache(2048) };
        WriteSequence(k, v, 768, ringRows: ring);

        long bytes = KvBlockTransfer.ComputeBlockByteSize(k, v, 256);
        var dst = new byte[bytes];
        Assert.True(KvBlockTransfer.Extract(_alloc, k, v, 768, 512, 256, dst, ringRows: ring));

        // Layer 0's K is the ring: positions 512..767 live at rows 0..255.
        var span = new ReadOnlySpan<byte>(dst);
        for (int h = 0; h < NumKVHeads; h++)
        {
            for (int i = 0; i < 256; i++)
            {
                for (int d = 0; d < HeadDim; d++)
                {
                    int idx = ((h * 256) + i) * HeadDim + d;
                    float got = BitConverter.ToSingle(span.Slice(idx * 4, 4));
                    Assert.Equal(Cell(0, h, 512 + i, d), got);
                }
            }
        }
    }

    /// <summary>
    /// A block whose positions the ring already overwrote cannot be reconstructed.
    /// Returning stale rows would be worse than refusing: the caller would cache them
    /// under the prefix hash and hand them to the next request.
    /// </summary>
    [Fact]
    public void Extract_BlockScrolledOutOfTheRing_ReturnsFalse()
    {
        const int ring = 512;
        var k = new[] { NewCache(ring), NewCache(2048) };
        var v = new[] { NewCache(ring), NewCache(2048) };
        WriteSequence(k, v, 1200, ringRows: ring);

        long bytes = KvBlockTransfer.ComputeBlockByteSize(k, v, 256);
        var dst = new byte[bytes];

        // Positions [0,256) are 1200 - 0 = 1200 behind the head; the ring holds 512.
        Assert.False(KvBlockTransfer.Extract(_alloc, k, v, 1200, 0, 256, dst, ringRows: ring));
        // The most recent block is still there.
        Assert.True(KvBlockTransfer.Extract(_alloc, k, v, 1200, 944, 256, dst, ringRows: ring));
    }

    [Fact]
    public void Extract_BlockWiderThanTheRing_ReturnsFalse()
    {
        const int ring = 128;
        var k = new[] { NewCache(ring), NewCache(2048) };
        var v = new[] { NewCache(ring), NewCache(2048) };
        WriteSequence(k, v, 512, ringRows: ring);

        long bytes = KvBlockTransfer.ComputeBlockByteSize(k, v, 256);
        var dst = new byte[bytes];
        Assert.False(KvBlockTransfer.Extract(_alloc, k, v, 512, 256, 256, dst, ringRows: ring));
    }

    // ------------------------------------------------------------------
    // Round trips
    // ------------------------------------------------------------------

    /// <summary>
    /// Capture every block as it fills, reset, replay the blocks in order, and the
    /// caches must be byte-identical to what the forward pass left - ring layers
    /// included. This is the property the prefix cache depends on.
    /// </summary>
    [Theory]
    [InlineData(512, 256, 1536)]   // block divides the ring
    [InlineData(640, 256, 1792)]   // block does NOT divide the ring: blocks straddle the wrap
    [InlineData(300, 100, 1000)]
    public void ExtractThenInject_RingAndLinearLayers_RoundTripsExactly(int ring, int blockSize, int seqLen)
    {
        var k = new[] { NewCache(ring), NewCache(2048), NewCache(ring) };
        var v = new[] { NewCache(ring), NewCache(2048), NewCache(ring) };
        WriteSequence(k, v, seqLen, ringRows: ring);
        byte[] expected = Snapshot(k, v);

        // Capture each block the moment it becomes full, which is when
        // BatchExecutor.CaptureNewlyFullBlocks runs.
        int fullBlocks = seqLen / blockSize;
        long blockBytes = KvBlockTransfer.ComputeBlockByteSize(k, v, blockSize);
        var captured = new byte[fullBlocks][];
        for (int b = 0; b < fullBlocks; b++)
        {
            captured[b] = new byte[blockBytes];
            int liveLen = (b + 1) * blockSize;   // the cache length when this block filled
            Assert.True(
                KvBlockTransfer.Extract(_alloc, k, v, liveLen, b * blockSize, blockSize, captured[b], ring),
                $"capture of block {b} failed");
        }

        // Fresh caches, then replay in order (what InjectAllBlocks does after a reset).
        var k2 = new[] { NewCache(ring), NewCache(2048), NewCache(ring) };
        var v2 = new[] { NewCache(ring), NewCache(2048), NewCache(ring) };
        for (int b = 0; b < fullBlocks; b++)
        {
            Assert.True(
                KvBlockTransfer.Inject(_alloc, k2, v2, b * blockSize, b * blockSize, blockSize, captured[b], ring),
                $"restore of block {b} failed");
        }
        // The tail the original wrote past the last full block is re-prefilled by the
        // caller, so replay it the same way to compare whole caches.
        for (int pos = fullBlocks * blockSize; pos < seqLen; pos++)
        {
            for (int l = 0; l < k2.Length; l++)
            {
                WritePosition(k2[l], 2 * l, pos, ring);
                WritePosition(v2[l], 2 * l + 1, pos, ring);
            }
        }

        Assert.Equal(expected, Snapshot(k2, v2));
    }

    /// <summary>
    /// Nothing about the linear path may change: uniform caches must produce exactly
    /// the bytes they did before ring support existed, whether or not a ring size is
    /// declared (no layer matches it).
    /// </summary>
    [Fact]
    public void Extract_UniformCache_IsUnaffectedByTheRingParameter()
    {
        var k = new[] { NewCache(1024), NewCache(1024) };
        var v = new[] { NewCache(1024), NewCache(1024) };
        WriteSequence(k, v, 700, ringRows: 0);

        long bytes = KvBlockTransfer.ComputeBlockByteSize(k, v, 256);
        var a = new byte[bytes];
        var b = new byte[bytes];
        Assert.True(KvBlockTransfer.Extract(_alloc, k, v, 700, 256, 256, a, ringRows: 0));
        Assert.True(KvBlockTransfer.Extract(_alloc, k, v, 700, 256, 256, b, ringRows: 512));
        Assert.Equal(a, b);
    }

    /// <summary>The linear guard also protects a plain overrun: a 256-token block at
    /// position 900 of a 1024-row cache must be refused, not clipped.</summary>
    [Fact]
    public void Extract_RangePastAUniformCachesRows_ReturnsFalse()
    {
        var k = new[] { NewCache(1024) };
        var v = new[] { NewCache(1024) };
        WriteSequence(k, v, 1024, ringRows: 0);

        long bytes = KvBlockTransfer.ComputeBlockByteSize(k, v, 256);
        var dst = new byte[bytes];
        // currentSeqLen is a lie here (the caller's bookkeeping drifted); the tensor
        // rows are the ground truth and they say no.
        Assert.False(KvBlockTransfer.Extract(_alloc, k, v, 4096, 900, 256, dst, ringRows: 0));
    }

    [Fact]
    public void Inject_BlockWiderThanTheRing_ReturnsFalse()
    {
        const int ring = 128;
        var k = new[] { NewCache(ring), NewCache(2048) };
        var v = new[] { NewCache(ring), NewCache(2048) };
        long bytes = KvBlockTransfer.ComputeBlockByteSize(k, v, 256);
        Assert.False(KvBlockTransfer.Inject(_alloc, k, v, 0, 0, 256, new byte[bytes], ringRows: ring));
    }

    // ------------------------------------------------------------------
    // Rewinding a ring (radix design M0c, P23)
    //
    // A ring layer keeps position p at row p % rows, so after the sequence passes
    // `rows` tokens every new position overwrites an old one. Truncation only moves
    // the head; the rows a wrap overwrote do not come back, and the next query still
    // attends a whole window behind the new head. Muse-Glimmer's truncation had no
    // guard, so a rewind deeper than the ring's slack (rows - window - 1) continued
    // from overwritten rows. Real geometry: window 2048, prefill chunk 2048, ring
    // pad256(2048 + 2048 + 1) = 4352 rows, slack 2303.
    // ------------------------------------------------------------------

    private const int GlimmerWindow = 2048;
    private const int GlimmerRing = 4352;
    private const int GlimmerSlack = GlimmerRing - GlimmerWindow - 1;   // 2303

    [Theory]
    // Never wrapped: every row is still there, any depth is exact.
    [InlineData(4000, 100, true)]
    [InlineData(GlimmerRing, 1, true)]
    // Wrapped: exact within the slack, including the executor's 16-token live rewind...
    [InlineData(6000, 6000 - 16, true)]
    [InlineData(6000, 6000 - GlimmerSlack, true)]
    // ...and refused one token past it.
    [InlineData(6000, 6000 - GlimmerSlack - 1, false)]
    [InlineData(20000, 5000, false)]
    // Dropping to nothing or to the head reads no old row.
    [InlineData(6000, 0, true)]
    [InlineData(6000, 6000, true)]
    // A target past the head is not a truncation.
    [InlineData(6000, 6001, false)]
    public void MuseGlimmer_CanTruncate_HonoursTheRingSlack(int cached, int target, bool expected)
    {
        var model = RingModel(cached);
        Assert.Equal(expected, model.CanTruncateKVCache(cached, target));
    }

    [Fact]
    public void MuseGlimmer_UniformCache_KeepsTheBaseRule()
    {
        // No ring (ggml-cpu, TP, or TS_MUSE_GLIMMER_SWA_RING=0): a linear cache rewinds
        // to any depth, as before.
        var model = RingModel(cachedTokens: 20000, ringRows: 0);
        Assert.True(model.CanTruncateKVCache(20000, 5));
        Assert.True(model.TryTruncateKVCache(5));
        Assert.Equal(5, model.CacheSeqLen);
    }

    [Fact]
    public void MuseGlimmer_TryTruncate_RefusesADeepRewindOnAWrappedRing_AndChangesNothing()
    {
        var model = RingModel(cachedTokens: 6000);
        Assert.False(model.TryTruncateKVCache(6000 - GlimmerSlack - 1));
        Assert.Equal(6000, model.CacheSeqLen);
    }

    [Fact]
    public void MuseGlimmer_TryTruncate_RewindsWithinTheSlack()
    {
        var model = RingModel(cachedTokens: 6000);
        Assert.True(model.TryTruncateKVCache(6000 - 16));
        Assert.Equal(6000 - 16, model.CacheSeqLen);
    }

    [Fact]
    public void MuseGlimmer_Truncate_ThrowsInsteadOfContinuingFromOverwrittenRows()
    {
        var model = RingModel(cachedTokens: 6000);
        var refused = Assert.Throws<InvalidOperationException>(() => model.TruncateKVCache(1000));
        Assert.Contains("TryTruncateKVCache", refused.Message);
        Assert.Equal(6000, model.CacheSeqLen);
    }

    /// <summary>A Muse-Glimmer instance with only the ring geometry and the head set:
    /// what the truncation guard reads. No tensors, so the cache-invalidation tail of
    /// an accepted rewind has nothing to touch.</summary>
    private static MuseGlimmerModel RingModel(int cachedTokens, int ringRows = GlimmerRing)
    {
        var model = (MuseGlimmerModel)System.Runtime.CompilerServices.RuntimeHelpers
            .GetUninitializedObject(typeof(MuseGlimmerModel));
        SetField(typeof(ModelBase), model, "<Config>k__BackingField",
            new TensorSharp.Runtime.ModelConfig { Architecture = "muse-glimmer", NumLayers = 2 });
        SetField(typeof(ModelBase), model, "<ExecutionPlan>k__BackingField", new BackendExecutionPlan(BackendType.Cpu));
        SetField(typeof(ModelBase), model, "_backend", BackendType.Cpu);
        SetField(typeof(ModelBase), model, "_cacheSeqLen", cachedTokens);
        SetField(typeof(MuseGlimmerModel), model, "_kvSwaRows", ringRows);
        SetField(typeof(MuseGlimmerModel), model, "_slidingWindow", GlimmerWindow);
        return model;
    }

    private static void SetField(Type owner, object target, string name, object value)
    {
        var field = owner.GetField(name,
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Public)
            ?? throw new MissingFieldException(owner.Name, name);
        field.SetValue(target, value);
    }
}
