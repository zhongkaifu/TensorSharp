// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace TensorSharp.Runtime;

/// <summary>Minimal streaming GGUF v3 writer: metadata key/values, then tensor descriptors whose data
/// callbacks write exactly their declared bytes in declaration order (data aligned to 32 bytes).</summary>
public sealed class GgufWriter
{
    public const uint TypeF32 = 0, TypeF16 = 1, TypeQ8_0 = 8;
    private const uint Alignment = 32;
    private readonly List<(string Key, Action<BinaryWriter> Value)> _metadata = new();
    private readonly List<(string Name, ulong[] Dims, uint Type, long Bytes, Action<Stream> Write)> _tensors = new();

    public void Add(string key, string value) => _metadata.Add((key, w => { w.Write(8u); WriteString(w, value); }));
    public void Add(string key, uint value) => _metadata.Add((key, w => { w.Write(4u); w.Write(value); }));
    public void Add(string key, int value) => _metadata.Add((key, w => { w.Write(5u); w.Write(value); }));
    public void Add(string key, float value) => _metadata.Add((key, w => { w.Write(6u); w.Write(value); }));
    public void Add(string key, bool value) => _metadata.Add((key, w => { w.Write(7u); w.Write(value ? (byte)1 : (byte)0); }));

    public void Add(string key, IReadOnlyList<string> values) => _metadata.Add((key, w =>
    {
        w.Write(9u); w.Write(8u); w.Write((ulong)values.Count);
        foreach (string v in values) WriteString(w, v);
    }));

    public void Add(string key, IReadOnlyList<int> values) => _metadata.Add((key, w =>
    {
        w.Write(9u); w.Write(5u); w.Write((ulong)values.Count);
        foreach (int v in values) w.Write(v);
    }));

    /// <summary>A tensor of ggml shape <paramref name="dims"/> (ne0 first); <paramref name="write"/> must
    /// write exactly <paramref name="bytes"/> bytes.</summary>
    public void AddTensor(string name, ulong[] dims, uint ggmlType, long bytes, Action<Stream> write)
        => _tensors.Add((name, dims, ggmlType, bytes, write));

    public static long Q8_0Bytes(long rows, long columns)
    {
        if (columns % 32 != 0) throw new ArgumentException("Q8_0 rows must hold whole 32-value blocks.", nameof(columns));
        return rows * (columns / 32) * 34;
    }

    public void WriteTo(string path)
    {
        using var file = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 22);
        using (var w = new BinaryWriter(file, Encoding.UTF8, leaveOpen: true))
        {
            w.Write(0x46554747u);   // "GGUF"
            w.Write(3u);
            w.Write((ulong)_tensors.Count);
            w.Write((ulong)_metadata.Count);
            foreach (var (key, value) in _metadata) { WriteString(w, key); value(w); }
            ulong offset = 0;
            foreach (var t in _tensors)
            {
                WriteString(w, t.Name);
                w.Write((uint)t.Dims.Length);
                foreach (ulong d in t.Dims) w.Write(d);
                w.Write(t.Type);
                w.Write(offset);
                offset = Align(offset + (ulong)t.Bytes);
            }
            w.Flush();
        }
        Pad(file);
        foreach (var t in _tensors)
        {
            long start = file.Position;
            t.Write(file);
            if (file.Position - start != t.Bytes)
                throw new InvalidDataException($"Tensor {t.Name} wrote {file.Position - start} bytes, declared {t.Bytes}.");
            Pad(file);
        }
    }

    private static ulong Align(ulong x) => (x + Alignment - 1) / Alignment * Alignment;

    private static void Pad(Stream s)
    {
        long pad = (long)(Align((ulong)s.Position) - (ulong)s.Position);
        if (pad > 0) s.Write(new byte[pad]);
    }

    private static void WriteString(BinaryWriter w, string s)
    {
        byte[] b = Encoding.UTF8.GetBytes(s);
        w.Write((ulong)b.Length);
        w.Write(b);
    }
}
