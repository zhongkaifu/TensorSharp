// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Text;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class GgufGeometryReadTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-geometry-" + Guid.NewGuid());

    public GgufGeometryReadTests() => Directory.CreateDirectory(_directory);

    [Fact]
    public void GeometryReaderPreservesSplitTensorRegionsWithoutDuplicateTokenizerData()
    {
        string first = WriteShard(1), second = WriteShard(2);
        using var full = new GgufFile(first);
        using var geometry = GgufFile.OpenForMemoryPlanning(first);
        Assert.Equal(full.DataOffset, geometry.DataOffset);
        Assert.Equal(full.FilePaths, geometry.FilePaths);
        Assert.Equal(2, geometry.Tensors.Count);
        Assert.Equal("example", geometry.GetString("general.architecture"));
        Assert.Equal(new[] { 1, 4, 7 }, geometry.GetInt32Array("example.attention.head_count_kv"));
        Assert.Equal(3, full.Metadata.Keys.Count(k => k.StartsWith("tokenizer.")));
        Assert.DoesNotContain(geometry.Metadata.Keys, k => k.StartsWith("tokenizer."));
        foreach (string name in full.Tensors.Keys)
        {
            Assert.Equal(full.GetTensorFileRegion(name), geometry.GetTensorFileRegion(name));
            Assert.Equal(full.ReadTensorData(full.Tensors[name]), geometry.ReadTensorData(geometry.Tensors[name]));
        }
        Assert.Equal(Path.GetFullPath(second), geometry.GetTensorFileRegion("weight-2").Path);
    }

    [Theory]
    [InlineData(GgufValueType.String, 100ul, GgufValueType.String)]
    [InlineData(GgufValueType.String, ulong.MaxValue, GgufValueType.String)]
    [InlineData(GgufValueType.Array, 100ul, GgufValueType.Float32)]
    [InlineData(GgufValueType.Array, 100ul, GgufValueType.String)]
    [InlineData(GgufValueType.Array, 0ul, GgufValueType.Array)]
    public void SkippedMetadataCannotSeekPastTruncationOrHideUnsupportedLayouts(
        GgufValueType type, ulong length, GgufValueType element)
    {
        string path = Path.Combine(_directory, "broken.gguf");
        using (var w = new BinaryWriter(File.Create(path)))
        {
            Header(w, 0, 1);
            Text(w, "tokenizer.ggml.tokens"); w.Write((uint)type);
            if (type == GgufValueType.Array) w.Write((uint)element);
            w.Write(length);
        }
        var error = Record.Exception(() => { using var ignored = GgufFile.OpenForMemoryPlanning(path); });
        Assert.NotNull(error);
        Assert.True(error is EndOfStreamException or InvalidDataException or NotSupportedException, error.ToString());
        // A failed constructor must not retain the file handle.
        using var exclusive = File.Open(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
    }

    private string WriteShard(int shard)
    {
        string path = Path.Combine(_directory, $"model-{shard:00000}-of-00002.gguf");
        using var w = new BinaryWriter(File.Create(path));
        Header(w, 1, 7);
        Text(w, "split.count"); w.Write((uint)GgufValueType.Uint16); w.Write((ushort)2);
        Text(w, "split.no"); w.Write((uint)GgufValueType.Uint16); w.Write((ushort)(shard - 1));
        Text(w, "tokenizer.ggml.tokens"); w.Write((uint)GgufValueType.Array);
        w.Write((uint)GgufValueType.String); w.Write(3ul); Text(w, ""); Text(w, "你好"); Text(w, "word");
        Text(w, "tokenizer.ggml.scores"); w.Write((uint)GgufValueType.Array);
        w.Write((uint)GgufValueType.Float32); w.Write(3ul); w.Write(0f); w.Write(1f); w.Write(2f);
        Text(w, "tokenizer.chat_template"); w.Write((uint)GgufValueType.String); Text(w, "{{ messages }}");
        Text(w, "general.architecture"); w.Write((uint)GgufValueType.String); Text(w, "example");
        Text(w, "example.attention.head_count_kv"); w.Write((uint)GgufValueType.Array);
        w.Write((uint)GgufValueType.Int32); w.Write(3ul); w.Write(1); w.Write(4); w.Write(7);
        Text(w, $"weight-{shard}"); w.Write(1u); w.Write(4ul); w.Write((uint)GgmlTensorType.F32); w.Write(0ul);
        while (w.BaseStream.Position % 32 != 0) w.Write((byte)0);
        for (int i = 0; i < 4; i++) w.Write((float)(shard * 10 + i));
        return path;
    }

    private static void Header(BinaryWriter w, ulong tensors, ulong metadata)
    { w.Write(0x46554747u); w.Write(3u); w.Write(tensors); w.Write(metadata); }
    private static void Text(BinaryWriter w, string text)
    { byte[] utf8 = Encoding.UTF8.GetBytes(text); w.Write((ulong)utf8.Length); w.Write(utf8); }
    public void Dispose() => Directory.Delete(_directory, recursive: true);
}
