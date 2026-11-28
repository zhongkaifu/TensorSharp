// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// LTX 2.5's AV denoiser weights as graph leaves: every projection a pointer into the GGUF's mapping in its stored format (the
// device caches key on it), every norm weight, bias, modulation table and the keyframe embedding an F32 tensor on the
// executor's allocator (BF16 and F16 widened exactly, as the direct denoiser held them); under whole-block stages
// (--layer-split N) each stage's executor gets its own copies of the small tensors its program binds, and under tensor
// parallelism (--tp N) each rank its shards of the block projections (Shard) and its slices of their biases and norms.
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

/// <summary>How a projection is placed over a tensor-parallel group (<c>--tp N</c>, Megatron sharding by attention heads and
/// feed-forward width).</summary>
internal enum Ltx25Parallel
{
    /// <summary>Whole on every rank: the patchify projections, the AdaLN MLPs and the output heads, which every rank computes.</summary>
    Replicated,
    /// <summary>A rank's output rows (its heads of q, k, v and the gate logits; its slice of the feed-forward's up projection),
    /// with the same slice of the bias.</summary>
    ColumnParallel,
    /// <summary>A rank's input columns (its heads of the attention's output projection; its slice of the feed-forward's down
    /// projection): the ranks' products are partial sums of the whole one, summed by an all-reduce before the bias.</summary>
    RowParallel,
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
                if (!IsProjection(name, info, Config)) continue;
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

    /// <summary>Whether tensor <paramref name="name"/> is a projection a step reads (a matrix named *.weight): not the prompt
    /// connectors' (the conditioner's), the modulation tables or the keyframe embedding (vectors to a step), nor the norms'
    /// weights (one axis).</summary>
    private static bool IsProjection(string name, GgufTensorInfo info, Ltx25Config config) =>
        name.StartsWith(config.Prefix, StringComparison.Ordinal) && name.EndsWith(".weight", StringComparison.Ordinal)
        && info.Shape.Length == 2 && !name.Contains("_embeddings_connector.", StringComparison.Ordinal)
        && !name.Contains("scale_shift_table", StringComparison.Ordinal) && !name.Contains("keyframes_abs_pos_embedding", StringComparison.Ordinal);

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

    // ---- tensor parallelism (--tp N)

    /// <summary>How projection <paramref name="relative"/> is placed over a tensor-parallel group: every attention of a block
    /// (self, text and cross-stream, of both streams) by heads, q, k, v and the head gates' logits column-parallel and to_out
    /// row-parallel; each feed-forward's up projection column-parallel and its down projection row-parallel; every projection
    /// outside the blocks replicated.</summary>
    internal static Ltx25Parallel ParallelOf(string relative)
    {
        if (!relative.StartsWith("transformer_blocks.", StringComparison.Ordinal)) return Ltx25Parallel.Replicated;
        if (relative.EndsWith(".to_out.0", StringComparison.Ordinal) || relative.EndsWith("ff.net.2", StringComparison.Ordinal))
            return Ltx25Parallel.RowParallel;
        if (relative.EndsWith(".to_q", StringComparison.Ordinal) || relative.EndsWith(".to_k", StringComparison.Ordinal)
            || relative.EndsWith(".to_v", StringComparison.Ordinal) || relative.EndsWith(".to_gate_logits", StringComparison.Ordinal)
            || relative.EndsWith("ff.net.0.proj", StringComparison.Ordinal))
            return Ltx25Parallel.ColumnParallel;
        return Ltx25Parallel.Replicated;
    }

    /// <summary>
    /// Why the blocks cannot shard over <paramref name="ranks"/> devices, or null: every rank holds whole heads of every
    /// attention (the video and audio heads divide by the ranks), every column-parallel projection's output rows split evenly,
    /// and every row-parallel projection's input columns split on whole quant blocks (each rank's columns are copied row by row,
    /// so a rank's share of a row cannot start inside a block: a Q4_K row of 2,048 values, eight 256-value blocks, splits over
    /// at most eight GPUs).
    /// </summary>
    internal static string TensorParallelRefusal(int videoHeads, int audioHeads,
        IEnumerable<(string Name, GgmlTensorType Type, long Ne0, long Ne1)> projections, int ranks)
    {
        ArgumentNullException.ThrowIfNull(projections);
        if (ranks < 2) return null;
        if (videoHeads % ranks != 0 || audioHeads % ranks != 0)
            return $"LTX 2.5 tensor parallelism needs a GPU count that divides its {videoHeads} video and {audioHeads} audio attention " +
                $"heads (every GPU holds whole heads); --tp {ranks} does not.";
        foreach (var (name, type, ne0, ne1) in projections)
        {
            Ltx25Parallel placement = ParallelOf(name);
            if (placement == Ltx25Parallel.ColumnParallel && ne1 % ranks != 0)
                return $"LTX 2.5 tensor parallelism over {ranks} GPUs cannot split {name}'s {ne1} output rows evenly.";
            long block = GgufFile.GetBlockSize(type);
            if (placement == Ltx25Parallel.RowParallel && ne0 % (block * ranks) != 0)
                return $"LTX 2.5 tensor parallelism over {ranks} GPUs would cut the {type} rows of {name} ({ne0} values, {ne0 / block} " +
                    $"blocks of {block}) inside a quant block: each GPU takes whole blocks of its input columns. Use a --tp that divides {ne0 / block}.";
        }
        return null;
    }

    /// <summary><see cref="TensorParallelRefusal(int, int, IEnumerable{ValueTuple{string, GgmlTensorType, long, long}}, int)"/>
    /// over the projections of <paramref name="file"/> (a model checks its checkpoint before it reads a weight).</summary>
    internal static string TensorParallelRefusal(GgufFile file, Ltx25Config config, int ranks)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(config);
        return TensorParallelRefusal(config.VideoHeads, config.AudioHeads, file.Tensors
            .Where(t => IsProjection(t.Key, t.Value, config))
            .Select(t => (t.Key[config.Prefix.Length..^".weight".Length], t.Value.Type, checked((long)t.Value.Shape[0]),
                checked((long)t.Value.Shape[1]))), ranks);
    }

    /// <summary>
    /// The block projections each of <paramref name="ranks"/> ranks binds, by name (<see cref="ParallelOf"/>; the replicated
    /// ones are not in it): a column-parallel projection's rank rows are a view of the mapping (its pages, its file offset), a
    /// row-parallel one's input columns a block-aligned copy per row, owned by <paramref name="owned"/> (freed with the
    /// denoiser; such a copy is read from host memory, never from the file again). Throws where the blocks do not shard
    /// (<see cref="TensorParallelRefusal(int, int, IEnumerable{ValueTuple{string, GgmlTensorType, long, long}}, int)"/>).
    /// </summary>
    public Dictionary<string, Ltx25Weight>[] Shard(int ranks, List<IntPtr> owned)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(owned);
        if (TensorParallelRefusal(Config.VideoHeads, Config.AudioHeads,
                _projections.Values.Select(w => (w.Name, (GgmlTensorType)w.Format, w.Ne0, w.Ne1)), ranks) is string refusal)
            throw new NotSupportedException(refusal);
        var shards = new Dictionary<string, Ltx25Weight>[ranks];
        for (int r = 0; r < ranks; r++) shards[r] = new Dictionary<string, Ltx25Weight>(StringComparer.Ordinal);
        foreach (Ltx25Weight w in _projections.Values)
        {
            Ltx25Parallel placement = ParallelOf(w.Name);
            if (placement == Ltx25Parallel.Replicated) continue;
            var whole = new DitShards.Slice(w.Format, w.Ne0, w.Ne1, w.Data, w.Bytes);
            for (int r = 0; r < ranks; r++)
            {
                if (placement == Ltx25Parallel.ColumnParallel)
                {
                    DitShards.Slice rows = DitShards.Rows(whole, 1, r, ranks, owned);
                    long offset = rows.Data - w.Data;
                    shards[r][w.Name] = w with { Ne1 = rows.Ne1, Data = rows.Data, Bytes = rows.Bytes, FileOffset = w.FileOffset < 0 ? -1 : w.FileOffset + offset };
                }
                else
                {
                    DitShards.Slice columns = DitShards.Columns(whole, r, ranks, owned);
                    shards[r][w.Name] = w with { Ne0 = columns.Ne0, Data = columns.Data, Bytes = columns.Bytes, FileOffset = -1 };
                }
            }
        }
        return shards;
    }

    /// <summary>Rank <paramref name="rank"/> of <paramref name="ranks"/>'s slice of the small vector <paramref name="name"/> (a
    /// column-parallel projection's bias, a whole-inner q or k norm's weight: its heads' values) as F32 on
    /// <paramref name="allocator"/>, made once a rank.</summary>
    public Tensor VectorSlice(string name, int rank, int ranks, IAllocator allocator)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(allocator);
        if (!_vectors.TryGetValue(allocator, out Dictionary<string, Tensor> made))
            _vectors[allocator] = made = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        string key = $"{name}#{rank}/{ranks}";
        if (made.TryGetValue(key, out Tensor t)) return t;
        float[] slice = DitShards.Vector(DirectOps.DequantTensor(_file, Config.Prefix + name), 1, rank, ranks);
        t = new Tensor(allocator, DType.Float32, slice.Length);
        t.SetElementsAsFloat(slice);
        made[key] = t;
        return t;
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
