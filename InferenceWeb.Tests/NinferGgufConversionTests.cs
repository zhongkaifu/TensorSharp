// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Buffers.Binary;
using System.IO;
using System.Linq;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class NinferGgufConversionTests
{
    [Fact]
    public void GgufWriterOutputReadsBackThroughGgufFile()
    {
        string path = Path.Combine(Path.GetTempPath(), $"tsg-writer-{Guid.NewGuid():N}.gguf");
        try
        {
            var w = new GgufWriter();
            w.Add("general.architecture", "qwen35");
            w.Add("qwen35.block_count", 65u);
            w.Add("qwen35.rope.freq_base", 1e7f);
            w.Add("tokenizer.ggml.tokens", new[] { "a", "Ġb", "<|im_end|>" });
            w.Add("tokenizer.ggml.token_type", new[] { 1, 1, 3 });
            var vector = new[] { 1.5f, -2f, 0.25f };
            w.AddTensor("v", new[] { 3UL }, GgufWriter.TypeF32, 12, s => { foreach (float f in vector) s.Write(BitConverter.GetBytes(f)); });
            w.AddTensor("m", new[] { 64UL, 2UL }, GgufWriter.TypeQ8_0, GgufWriter.Q8_0Bytes(2, 64), s => s.Write(new byte[GgufWriter.Q8_0Bytes(2, 64)]));
            w.WriteTo(path);

            using var g = new GgufFile(path);
            Assert.Equal("qwen35", g.GetString("general.architecture"));
            Assert.Equal(new[] { "a", "Ġb", "<|im_end|>" }, g.GetStringArray("tokenizer.ggml.tokens"));
            Assert.Equal(new ulong[] { 64, 2 }, g.Tensors["m"].Shape);
            Assert.Equal(new ulong[] { 3 }, g.Tensors["v"].Shape);
            byte[] bytes = File.ReadAllBytes(path);
            long at = g.DataOffset + (long)g.Tensors["v"].Offset;
            Assert.Equal(-2f, BitConverter.ToSingle(bytes, (int)at + 4));
            Assert.Equal(0, (g.DataOffset + (long)g.Tensors["m"].Offset) % 32);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void Q8_0BlocksFollowGgmlsReferenceRule()
    {
        var x = Enumerable.Range(0, 32).Select(i => (i - 16) * 0.5f).ToArray();   // amax 8 -> d = 8 / 127
        var y = new byte[34];
        NinferQwen38GgufConverter.QuantizeQ8_0(x, y);
        float d = (float)BinaryPrimitives.ReadHalfLittleEndian(y);
        Assert.Equal((float)(System.Half)(8f / 127f), d);
        for (int i = 0; i < 32; i++)
            Assert.Equal((sbyte)MathF.Round(x[i] * (127f / 8f), MidpointRounding.AwayFromZero), (sbyte)y[2 + i]);
        Assert.Equal(-127, (sbyte)y[2]);
    }

    [Fact]
    public void GdnValueHeadOrderIsAPermutationGroupedByKeyHead()
    {
        // GGUF head i = artifact head (i % 16) * 3 + i / 16 (checked against the same model's Q8_0 GGUF).
        var map = Enumerable.Range(0, 48).Select(i => i % 16 * 3 + i / 16).ToArray();
        Assert.Equal(Enumerable.Range(0, 48), map.OrderBy(v => v));
        Assert.Equal(new[] { 0, 3, 6 }, map.Take(3));
        Assert.Equal(1, map[16]);
        Assert.Equal(47, map[47]);
    }
}
