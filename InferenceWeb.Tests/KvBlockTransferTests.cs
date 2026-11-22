// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Regression tests for the KV block snapshot helper (KvBlockTransfer).
//
// The bug these pin down: the legacy Muse-Glimmer model (removed in stage S9 of
// docs/design/muse-glimmer-unified.md) sized its 39 sliding-window layers shorter than
// its full layers, and KvBlockTransfer addressed every layer as if it had the full
// context's rows, so the first block whose start passed the short layer's rows read
// straight past the end of that allocation. The server died with
//
//   System.AccessViolationException: Attempted to read or write protected memory.
//      at TensorSharp.Models.KvBlockTransfer.CopyOneOut(...)
//
// The helper validates every range against each tensor's own rows before a byte moves.
// The ring addressing the legacy model used went with it; these tests keep the linear
// path and the per-tensor guard. They run on the CPU allocator with no GGUF, so they are
// unconditional.
using System;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public class KvBlockTransferTests : IDisposable
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

    private Tensor NewCache(int rows)
    {
        var t = new Tensor(_alloc, DType.Float32, NumKVHeads, rows, HeadDim);
        Ops.Fill(t, 0f);
        _owned.Add(t);
        return t;
    }

    /// <summary>A value unique to (tensor, head, position, dim) so a misplaced copy is impossible to mistake for a
    /// correct one.</summary>
    private static float Cell(int tag, int head, int pos, int dim)
        => tag * 1_000_000f + head * 10_000f + pos * 10f + dim;

    /// <summary>Write positions [0, <paramref name="seqLen"/>) into every layer, as a forward pass would (row = position),
    /// stopping at each layer's own rows.</summary>
    private static void WriteSequence(Tensor[] k, Tensor[] v, int seqLen, int from = 0)
    {
        for (int l = 0; l < k.Length; l++)
            foreach (var (t, tag) in new[] { (k[l], 2 * l), (v[l], 2 * l + 1) })
            {
                int rows = (int)t.Sizes[1];
                for (int pos = from; pos < Math.Min(seqLen, rows); pos++)
                    for (int h = 0; h < NumKVHeads; h++)
                        for (int d = 0; d < HeadDim; d++)
                            t.SetElementAsFloat(Cell(tag, h, pos, d), h, pos, d);
            }
    }

    private static byte[] Snapshot(Tensor[] k, Tensor[] v)
    {
        var bytes = new System.Collections.Generic.List<byte>();
        for (int l = 0; l < k.Length; l++)
            foreach (var t in new[] { k[l], v[l] })
            {
                int rows = (int)t.Sizes[1];
                for (int h = 0; h < NumKVHeads; h++)
                    for (int r = 0; r < rows; r++)
                        for (int d = 0; d < HeadDim; d++)
                            bytes.AddRange(BitConverter.GetBytes(t.GetElementAsFloat(h, r, d)));
            }
        return bytes.ToArray();
    }

    /// <summary>The shape that faulted: a 512-row layer next to a 2048-row one, asked for the block starting at 512.
    /// The short layer cannot address it, so the block is refused, not attempted.</summary>
    [Fact]
    public void Extract_BlockPastAShortLayersRows_IsRefused()
    {
        var k = new[] { NewCache(512), NewCache(2048) };
        var v = new[] { NewCache(512), NewCache(2048) };
        WriteSequence(k, v, 768);
        var dst = new byte[KvBlockTransfer.ComputeBlockByteSize(k, v, 256)];
        Assert.False(KvBlockTransfer.Extract(_alloc, k, v, 768, 512, 256, dst));
    }

    /// <summary>A plain overrun: a 256-token block at position 900 of a 1024-row cache is refused, not clipped, even when
    /// the caller's sequence length says it is there (the tensor's rows are the ground truth).</summary>
    [Fact]
    public void Extract_RangePastTheCachesRows_ReturnsFalse()
    {
        var k = new[] { NewCache(1024) };
        var v = new[] { NewCache(1024) };
        WriteSequence(k, v, 1024);
        var dst = new byte[KvBlockTransfer.ComputeBlockByteSize(k, v, 256)];
        Assert.False(KvBlockTransfer.Extract(_alloc, k, v, 4096, 900, 256, dst));
    }

    /// <summary>An injected block past a layer's rows is refused too, and writes nothing.</summary>
    [Fact]
    public void Inject_BlockPastAShortLayersRows_ReturnsFalse()
    {
        var k = new[] { NewCache(300), NewCache(2048) };
        var v = new[] { NewCache(300), NewCache(2048) };
        byte[] before = Snapshot(k, v);
        var block = new byte[KvBlockTransfer.ComputeBlockByteSize(k, v, 256)];
        Assert.False(KvBlockTransfer.Inject(_alloc, k, v, 256, 256, 256, block));
        Assert.Equal(before, Snapshot(k, v));
    }

    /// <summary>Capture every block as it fills, reset, replay the blocks in order, and the caches are byte-identical to
    /// what the forward pass left: the property the prefix cache depends on.</summary>
    [Theory]
    [InlineData(256, 1536)]
    [InlineData(100, 1000)]
    public void ExtractThenInject_RoundTripsExactly(int blockSize, int seqLen)
    {
        var k = new[] { NewCache(2048), NewCache(2048) };
        var v = new[] { NewCache(2048), NewCache(2048) };
        WriteSequence(k, v, seqLen);
        byte[] expected = Snapshot(k, v);

        int fullBlocks = seqLen / blockSize;
        long blockBytes = KvBlockTransfer.ComputeBlockByteSize(k, v, blockSize);
        var captured = new byte[fullBlocks][];
        for (int b = 0; b < fullBlocks; b++)
        {
            captured[b] = new byte[blockBytes];
            Assert.True(KvBlockTransfer.Extract(_alloc, k, v, (b + 1) * blockSize, b * blockSize, blockSize, captured[b]),
                $"capture of block {b} failed");
        }

        var k2 = new[] { NewCache(2048), NewCache(2048) };
        var v2 = new[] { NewCache(2048), NewCache(2048) };
        for (int b = 0; b < fullBlocks; b++)
            Assert.True(KvBlockTransfer.Inject(_alloc, k2, v2, b * blockSize, b * blockSize, blockSize, captured[b]),
                $"restore of block {b} failed");
        // The tail past the last full block is re-prefilled by the caller.
        WriteSequence(k2, v2, seqLen, from: fullBlocks * blockSize);
        Assert.Equal(expected, Snapshot(k2, v2));
    }

    /// <summary>Blocks are appended in order only: an inject that does not start at the cache's length is refused.</summary>
    [Fact]
    public void Inject_OutOfOrder_ReturnsFalse()
    {
        var k = new[] { NewCache(1024) };
        var v = new[] { NewCache(1024) };
        var block = new byte[KvBlockTransfer.ComputeBlockByteSize(k, v, 128)];
        Assert.False(KvBlockTransfer.Inject(_alloc, k, v, 0, 128, 128, block));
    }
}
