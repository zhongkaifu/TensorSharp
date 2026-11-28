// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// LTX 2.5's AV denoiser weights as graph leaves: every projection a pointer into the GGUF's mapping in its stored format (the
// device caches key on it), every norm weight, bias, modulation table and the keyframe embedding an F32 tensor on the
// executor's allocator (BF16 and F16 widened exactly, as the direct denoiser held them); under whole-block stages
// (--layer-split N) each stage's executor gets its own copies of the small tensors its program binds.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp.Graph;
using TensorSharp.Models.Direct;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Ltx25;

/// <summary>A projection leaf [Ne0 (in), Ne1 (out)] keyed by its host bytes: a tensor of the GGUF's mapping, or (an
/// Ingredients factor) F32 values the adapter set owns, whose products keep exact F32 arithmetic.</summary>
internal sealed record Ltx25Weight(string Name, int Format, long Ne0, long Ne1, IntPtr Data, long Bytes, long FileOffset = -1,
    bool Exact = false) : IGraphWeight
{
    public long Ne2 => 1;
    public IntPtr CacheKey => Data;
    public bool ExactProducts => Exact;
}

/// <summary>The tensors of an LTX 2.5 AV denoiser GGUF, bound for the denoiser's step programs.</summary>
internal sealed class Ltx25DenoiserWeights : IDisposable
{
    private readonly GgufFile _file;
    private readonly IAllocator _allocator;
    private readonly Dictionary<string, Ltx25Weight> _projections = new(StringComparer.Ordinal);
    // The small tensors made so far, by the allocator they live on (the executor's, or under stages each stage's).
    private readonly Dictionary<IAllocator, Dictionary<string, Tensor>> _vectors = new(ReferenceEqualityComparer.Instance);
    private bool _disposed;

    public Ltx25DenoiserWeights(string path, IAllocator allocator)
    {
        _allocator = allocator ?? throw new ArgumentNullException(nameof(allocator));
        _file = new GgufFile(path ?? throw new ArgumentNullException(nameof(path)));
        try
        {
            Config = Ltx25Config.FromGguf(_file);
            Path = System.IO.Path.GetFullPath(path);
            foreach (var (name, info) in _file.Tensors)
            {
                // The projections a step reads (matrices named *.weight): not the prompt connectors' (the conditioner's), the
                // modulation tables or the keyframe embedding (vectors to a step), nor the norms' weights (one axis).
                if (!name.StartsWith(Config.Prefix, StringComparison.Ordinal) || !name.EndsWith(".weight", StringComparison.Ordinal)
                    || info.Shape.Length != 2 || name.Contains("_embeddings_connector.", StringComparison.Ordinal)
                    || name.Contains("scale_shift_table", StringComparison.Ordinal) || name.Contains("keyframes_abs_pos_embedding", StringComparison.Ordinal))
                    continue;
                string relative = name[Config.Prefix.Length..^".weight".Length];
                if (!_file.TryGetTensorDataPointer(info, out IntPtr data))
                    throw new InvalidDataException($"LTX tensor '{name}' has no data pointer.");
                long ne0 = (long)info.Shape[0], ne1 = (long)info.Shape[1];
                _projections[relative] = new Ltx25Weight(relative, (int)info.Type, ne0, ne1, data, _file.GetTensorByteCount(info),
                    checked(_file.DataOffset + (long)info.Offset));
            }
            if (_projections.Count == 0) throw new InvalidDataException("The LTX denoiser GGUF holds no projections.");
        }
        catch
        {
            _file.Dispose();
            throw;
        }
    }

    public Ltx25Config Config { get; }
    public string Path { get; }

    /// <summary>The checkpoint's file (the CUDA uploads read it ahead).</summary>
    internal GgufFile File => _file;

    /// <summary>Every projection a step binds.</summary>
    public IReadOnlyCollection<Ltx25Weight> Projections => _projections.Values;

    /// <summary>The device bytes the projections take.</summary>
    public long Bytes => _projections.Values.Sum(w => w.Bytes);

    /// <summary>The projection <paramref name="name"/> (relative to the checkpoint's prefix, without ".weight").</summary>
    public Ltx25Weight Projection(string name) => _projections.TryGetValue(name, out Ltx25Weight w) ? w
        : throw new InvalidDataException($"The LTX denoiser has no projection '{name}'.");

    /// <summary>Whether the projection has a bias.</summary>
    public bool HasBias(string name) => _file.Tensors.ContainsKey(Config.Prefix + name + ".bias");

    /// <summary>A small tensor (a bias, a norm weight, a modulation table, the keyframe embedding) as F32 on the executor's
    /// allocator, in the checkpoint's row-major order, made once.</summary>
    public Tensor Vector(string name) => Vector(name, _allocator);

    /// <summary><see cref="Vector(string)"/> on <paramref name="allocator"/>: a whole-block stage's tensor executor binds the
    /// small tensors of its own program on its own device (another GPU's tensor is not its memory), so each stage device gets
    /// its copy the first time its program emits the tensor.</summary>
    public Tensor Vector(string name, IAllocator allocator)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(allocator);
        if (!_vectors.TryGetValue(allocator, out Dictionary<string, Tensor> made))
            _vectors[allocator] = made = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        if (made.TryGetValue(name, out Tensor t)) return t;
        string full = Config.Prefix + name;
        float[] values = DirectOps.DequantTensor(_file, full);
        // Sized as the ggml extents read outermost first (a [dim, 9] table is 9 rows of dim).
        t = new Tensor(allocator, DType.Float32, Extents(name).Reverse().ToArray());
        t.SetElementsAsFloat(values);
        made[name] = t;
        return t;
    }

    /// <summary>The ggml extents of a small tensor (innermost first).</summary>
    public long[] Extents(string name) => _file.Tensors[Config.Prefix + name].Shape.Select(s => checked((long)s)).ToArray();

    /// <summary>Each block's projection bytes, in block order: a whole-block split (<see cref="DenoiserStages"/>) balances
    /// them.</summary>
    public long[] BlockBytes()
    {
        var bytes = new long[Config.Layers];
        foreach (Ltx25Weight w in _projections.Values)
            if (BlockOf(w.Name) is int block and >= 0 && block < bytes.Length) bytes[block] += w.Bytes;
        return bytes;
    }

    /// <summary>Whether the program of <paramref name="span"/> binds projection <paramref name="w"/>: its blocks' projections;
    /// the patchify projections where it opens the step and the output heads where it closes it; and every other projection
    /// outside the blocks (the eight AdaLN MLPs), which every stage computes from the step's timestep features itself.</summary>
    public bool Binds(DenoiserSpan span, Ltx25Weight w)
    {
        int block = BlockOf(w.Name);
        if (block >= 0) return block >= span.First && block < span.End;
        if (w.Name is "patchify_proj" or "audio_patchify_proj") return span.IsFirst;
        if (w.Name is "proj_out" or "audio_proj_out") return span.IsLast;
        return true;
    }

    /// <summary>The projection bytes the program of <paramref name="span"/> binds (<see cref="Binds"/>): what its stage's
    /// device holds.</summary>
    public long BytesOf(DenoiserSpan span) => _projections.Values.Where(w => Binds(span, w)).Sum(w => w.Bytes);

    /// <summary>The block index of a projection name, or -1 for one outside the blocks (patchify, the AdaLN embedders, the
    /// output heads).</summary>
    internal int BlockOf(string relative)
    {
        const string prefix = "transformer_blocks.";
        if (!relative.StartsWith(prefix, StringComparison.Ordinal)) return -1;
        int end = relative.IndexOf('.', prefix.Length);
        return end > prefix.Length && int.TryParse(relative.AsSpan(prefix.Length, end - prefix.Length), out int b) ? b : -1;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (Dictionary<string, Tensor> made in _vectors.Values)
            foreach (Tensor t in made.Values) t.Dispose();
        _vectors.Clear();
        _projections.Clear();
        _file.Dispose();
    }
}
