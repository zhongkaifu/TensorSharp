// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.GGML;
using TensorSharp.Models.QwenImage21;

namespace InferenceWeb.Tests;

/// <summary>Weight sharding for Qwen-Image-2.1 tensor parallelism (QwenImage21Denoiser.ShardBlocks): which bytes each rank
/// receives, on a synthetic Q8_0 transformer. The sharded programs' numerics are compared with the unsharded ones by
/// eng/QwenImage21CompanionProbe dit-parity (two ranks of a loopback group on one device).</summary>
public sealed unsafe class QwenImage21TensorParallelShardTests : IDisposable
{
    private const int Q8Type = 8, Q8Block = 32, Q8Bytes = 34, Dim = 128, HeadDim = 16, Ff = 128, Layers = 2;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-qi21-shard-" + Guid.NewGuid().ToString("N"));

    public QwenImage21TensorParallelShardTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    /// <summary>The synthetic transformer of QwenImage21DenoiserTests at 128 wide (8 heads of 16), its block projections in
    /// Q8_0 (patterned bytes: the shards are compared, never run).</summary>
    private string Transformer(bool fused)
    {
        var tensors = new List<(string, long[], int, byte[])>();
        int seed = 1;
        void F32(string name, params long[] dims)
        {
            float[] v = QwenImage21DenoiserTests.Rand((int)dims.Aggregate(1L, (a, b) => a * b), seed++, 0.1f);
            var raw = new byte[v.Length * 4];
            Buffer.BlockCopy(v, 0, raw, 0, raw.Length);
            tensors.Add((name + ".weight", dims, 0, raw));
        }
        void Q8(string name, long input, long output)
        {
            var raw = new byte[input / Q8Block * Q8Bytes * output];
            for (int i = 0; i < raw.Length; i++) raw[i] = (byte)(i * 31 + seed * 7);
            seed++;
            tensors.Add((name + ".weight", new[] { input, output }, Q8Type, raw));
        }
        F32("img_in", 16, Dim);
        F32("txt_in.in_layer", 48, Dim);
        F32("txt_in.out_layer", Dim, Dim);
        F32("txt_in.text_norm", 48);
        F32("time_text_embed.timestep_embedder.linear_1", 256, Dim);
        F32("time_text_embed.timestep_embedder.linear_2", Dim, Dim);
        F32("modulation.1", Dim, 4 * Dim);
        F32("norm_out.linear", Dim, Dim);
        F32("proj_out", Dim, 16);
        for (int i = 0; i < Layers; i++)
        {
            string p = $"transformer_blocks.{i}.";
            Q8(p + "attn.to_q", Dim, Dim);
            Q8(p + "attn.to_k", Dim, Dim);
            Q8(p + "attn.to_v", Dim, Dim);
            Q8(p + "attn.to_out.0", Dim, Dim);
            if (fused) Q8(p + "img_mlp.gate_up", Dim, 2 * Ff);
            else
            {
                Q8(p + "img_mlp.gate_layer", Dim, Ff);
                Q8(p + "img_mlp.proj", Dim, Ff);
            }
            Q8(p + "img_mlp.out", Ff, Dim);
            F32(p + "attn.norm_q", HeadDim);
            F32(p + "attn.norm_k", HeadDim);
        }
        string path = Path.Combine(_dir, $"shard-{(fused ? "fused" : "separate")}.gguf");
        QwenImage21DenoiserTests.WriteGguf(path, tensors);
        return path;
    }

    private static byte[] Bytes(QwenImage21Weight w) => new ReadOnlySpan<byte>((void*)w.Data, (int)w.Bytes).ToArray();

    [Theory]
    [InlineData(true, 2)]
    [InlineData(false, 2)]
    [InlineData(true, 4)]
    public void RanksReceiveWholeHeadsAndMatchingMlpColumns(bool fused, int ranks)
    {
        using var w = new QwenImage21Weights(Transformer(fused));
        Assert.Equal((Dim, Dim / HeadDim, Ff, Layers), (w.Dim, w.Heads, w.Ff, w.Layers));
        var owned = new List<IntPtr>();
        try
        {
            var shards = QwenImage21Denoiser.ShardBlocks(w, ranks, owned);
            Assert.Equal(ranks, shards.Length);
            long rowBytes = Dim / Q8Block * Q8Bytes, local = Dim / ranks, ffLocal = Ff / ranks;
            for (int r = 0; r < ranks; r++)
                for (int layer = 0; layer < Layers; layer++)
                {
                    QwenImage21Block block = w.Blocks[layer], s = shards[r][layer];
                    // Column-parallel projections are views of consecutive output rows: whole heads, a slice of the MLP.
                    Assert.Equal(block.Q.Data + (nint)(r * local * rowBytes), s.Q.Data);
                    Assert.Equal((local, local * rowBytes), (s.K.Ne1, s.K.Bytes));
                    Assert.Equal(block.V.Data + (nint)(r * local * rowBytes), s.V.Data);
                    Assert.Equal(block.Gate.Data + (nint)(r * ffLocal * rowBytes), s.Gate.Data);
                    Assert.Equal(ffLocal, s.Gate.Ne1);
                    // Up comes after all gate rows in a fused [gate; up] projection.
                    var upSource = fused ? block.Gate.Data + (nint)(Ff * rowBytes) : block.Up.Data;
                    Assert.Equal(upSource + (nint)(r * ffLocal * rowBytes), s.Up.Data);
                    Assert.Equal(ffLocal, s.Up.Ne1);
                    Assert.False(s.FusedGateUp);
                    // Row-parallel projections copy this rank's input columns of every row.
                    AssertColumns(block.Out, s.Out, r, local);
                    AssertColumns(block.Down, s.Down, r, ffLocal);
                    Assert.Same(block.NormQ, s.NormQ);
                    Assert.Same(block.NormK, s.NormK);
                }
            Assert.Equal(2 * ranks * Layers, owned.Count);
        }
        finally { foreach (var p in owned) QuantizedWeight.FreeBuffer(p); }
    }

    private static void AssertColumns(QwenImage21Weight source, QwenImage21Weight shard, int rank, long columns)
    {
        Assert.Equal((columns, source.Ne1), (shard.Ne0, shard.Ne1));
        long sourceRow = source.Ne0 / Q8Block * Q8Bytes, row = columns / Q8Block * Q8Bytes;
        Assert.Equal(row * source.Ne1, shard.Bytes);
        byte[] all = Bytes(source), part = Bytes(shard);
        for (long o = 0; o < source.Ne1; o++)
            Assert.True(all.AsSpan((int)(o * sourceRow + rank * row), (int)row).SequenceEqual(part.AsSpan((int)(o * row), (int)row)),
                $"row {o} of rank {rank}");
    }

    [Fact]
    public void UnsplittableGroupsAreRefused()
    {
        using var w = new QwenImage21Weights(Transformer(true));
        var owned = new List<IntPtr>();
        try
        {
            // 8 heads cannot be divided over 3 GPUs.
            Assert.Throws<NotSupportedException>(() => QwenImage21Denoiser.ShardBlocks(w, 3, owned));
            // A 128-wide Q8_0 row has 4 blocks: 8 ranks would split a block.
            var error = Assert.Throws<NotSupportedException>(() => QwenImage21Denoiser.ShardBlocks(w, 8, owned));
            Assert.Contains("32-element blocks", error.Message);
        }
        finally { foreach (var p in owned) QuantizedWeight.FreeBuffer(p); }
    }
}
