// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using TensorSharp.Graph;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Ltx25;

/// <summary>
/// The LTX 2.5 Ingredients IC-LoRA as the denoiser's adapter set: publisher-scaled B @ A updates (no separate alpha / rank
/// multiplier) on the attention and feed-forward projections of the video stream's blocks, applied unmerged in exact F32 (the
/// base checkpoint is never requantized). Opening validates every factor pair against the checkpoint; each projection's
/// factors are read (widened exactly to F32) the first time a step emits it, and kept in pinned host memory whose address the
/// device copies key on.
/// </summary>
public sealed class Ltx25LoraSet : IDisposable
{
    private static long s_generation;
    private readonly SafetensorsFile _file;
    private readonly Dictionary<string, Pair> _pairs = new(StringComparer.Ordinal);
    private readonly Dictionary<string, (Ltx25Weight Down, Ltx25Weight Up)> _loaded = new(StringComparer.Ordinal);
    private readonly int _layers, _videoDim;
    private bool _disposed;

    private sealed record Pair(string A, string B, int Input, int Output, int Rank);

    private Ltx25LoraSet(string path, GgufFile checkpoint, Ltx25Config config)
    {
        Path = System.IO.Path.GetFullPath(path);
        _file = new SafetensorsFile(Path);
        _layers = config.Layers;
        _videoDim = config.VideoDim;
        Generation = Interlocked.Increment(ref s_generation);
        try
        {
            if (!_file.Metadata.TryGetValue("model_version", out string version) || version != "2.5" ||
                !_file.Metadata.TryGetValue("reference_downscale_factor", out string downscale) || downscale != "1")
                throw new InvalidDataException("The Ingredients adapter must declare model_version=2.5 and reference_downscale_factor=1.");
            foreach (var tensor in _file.Tensors.Values)
            {
                string name = tensor.Name;
                const string prefix = "diffusion_model.";
                if (name.StartsWith(prefix, StringComparison.Ordinal)) name = name[prefix.Length..];
                const string aSuffix = ".lora_A.weight", bSuffix = ".lora_B.weight";
                bool isA = name.EndsWith(aSuffix, StringComparison.Ordinal);
                if (!isA && !name.EndsWith(bSuffix, StringComparison.Ordinal))
                    throw new InvalidDataException("Unsupported Ingredients tensor: " + tensor.Name);
                string target = name[..^(isA ? aSuffix.Length : bSuffix.Length)];
                if (!ValidTarget(target, config.Layers))
                    throw new InvalidDataException("Unsupported Ingredients projection: " + target);
                if (tensor.Dtype is not (SafetensorDtype.BF16 or SafetensorDtype.F16 or SafetensorDtype.F32) ||
                    tensor.Shape.Length != 2 || tensor.Shape.Any(size => size <= 0 || size > int.MaxValue))
                    throw new InvalidDataException("Ingredients factors must be nonempty BF16/F16/F32 matrices: " + tensor.Name);
                if (!isA) continue;
                string bName = tensor.Name[..^aSuffix.Length] + bSuffix;
                if (!_file.Tensors.TryGetValue(bName, out var b))
                    throw new InvalidDataException("Missing Ingredients B factor for " + target);
                if (!checkpoint.Tensors.TryGetValue(config.Prefix + target + ".weight", out var weight) ||
                    weight.Shape.Length != 2 || b.Shape.Length != 2 ||
                    tensor.Shape[0] != b.Shape[1] || tensor.Shape[1] != (long)weight.Shape[0] || b.Shape[0] != (long)weight.Shape[1])
                    throw new InvalidDataException("Ingredients factors do not match the base projection: " + target);
                if (!_pairs.TryAdd(target, new(tensor.Name, bName, checked((int)tensor.Shape[1]),
                    checked((int)b.Shape[0]), checked((int)tensor.Shape[0]))))
                    throw new InvalidDataException("Duplicate Ingredients projection: " + target);
            }
            if (_pairs.Count == 0 || _file.Tensors.Count != checked(_pairs.Count * 2))
                throw new InvalidDataException("Every Ingredients projection requires exactly one A/B factor pair.");
            Adapter = new DiffusionAdapter { Name = System.IO.Path.GetFileName(Path), Payload = this };
            AdapterSet = new AdapterSet(new[] { Adapter });
        }
        catch
        {
            _file.Dispose();
            throw;
        }
    }

    /// <summary>Opens and validates the adapter at <paramref name="path"/> against the denoiser <paramref name="checkpoint"/>.</summary>
    public static Ltx25LoraSet Open(string path, GgufFile checkpoint, Ltx25Config config)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(checkpoint);
        ArgumentNullException.ThrowIfNull(config);
        return new Ltx25LoraSet(path, checkpoint, config);
    }

    /// <summary>Opens the adapter at <paramref name="path"/> against the denoiser GGUF at <paramref name="checkpoint"/>.</summary>
    public static Ltx25LoraSet Open(string path, string checkpoint)
    {
        using var file = new GgufFile(checkpoint);
        return Open(path, file, Ltx25Config.FromGguf(file));
    }

    public string Path { get; }

    /// <summary>A number no other set shares (a plan binds one set's factors).</summary>
    public long Generation { get; }

    public int ProjectionCount => _pairs.Count;

    /// <summary>Projections whose factors were read so far.</summary>
    internal int LoadedProjectionCount { get { lock (_loaded) return _loaded.Count; } }

    /// <summary>The set as one runtime adapter (its payload), and as the adapter set a reference clip runs with.</summary>
    public DiffusionAdapter Adapter { get; }
    public AdapterSet AdapterSet { get; }

    /// <summary>The F32 bytes of every factor once read (the device copies take as much).</summary>
    public long FactorBytes => _pairs.Values.Sum(p => (long)p.Rank * (p.Input + p.Output) * sizeof(float));

    /// <summary>The host addresses the factors' device copies are keyed by (those read so far).</summary>
    internal IReadOnlyList<IntPtr> FactorAddresses
    {
        get
        {
            lock (_loaded) return _loaded.Values.SelectMany(f => new[] { f.Down.Data, f.Up.Data }).ToArray();
        }
    }

    /// <summary>Whether the set was validated against a checkpoint of <paramref name="config"/>'s shape.</summary>
    internal bool Fits(Ltx25Config config) => config.Layers == _layers && config.VideoDim == _videoDim;

    /// <summary>The factors of <paramref name="projection"/> (down = A [in, rank], up = B [rank, out], F32, exact products), read
    /// on first use, or null where the adapter does not update it.</summary>
    internal (Ltx25Weight Down, Ltx25Weight Up)? Factors(string projection)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_pairs.TryGetValue(projection, out Pair pair)) return null;
        lock (_loaded)
            if (_loaded.TryGetValue(projection, out var cached)) return cached;
        Ltx25Weight Read(string tensor, long ne0, long ne1, string label)
        {
            float[] values = _file.ReadFloat32(tensor);
            if (values.Any(value => !float.IsFinite(value)))
                throw new InvalidDataException($"Ingredients {label} factor contains nonfinite values: " + projection);
            long bytes = values.LongLength * sizeof(float);
            IntPtr host = Marshal.AllocHGlobal((IntPtr)bytes);
            Marshal.Copy(values, 0, host, values.Length);
            return new Ltx25Weight(projection + "." + label, GraphFormat.F32, ne0, ne1, host, bytes, Exact: true);
        }
        Ltx25Weight down = Read(pair.A, pair.Input, pair.Rank, "A");
        Ltx25Weight up;
        try { up = Read(pair.B, pair.Rank, pair.Output, "B"); }
        catch { Marshal.FreeHGlobal(down.Data); throw; }
        lock (_loaded)
        {
            // A concurrent read of the same projection (the background preload) got there first: keep its copy.
            if (!_disposed && _loaded.TryAdd(projection, (down, up))) return (down, up);
            Marshal.FreeHGlobal(down.Data);
            Marshal.FreeHGlobal(up.Data);
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _loaded[projection];
        }
    }

    private System.Threading.Tasks.Task _preload;

    /// <summary>Read every factor in the background (a reference clip's text encode overlaps the adapter's 1.3 GB read, which a
    /// first step otherwise waits for); a step that needs a factor before it is read reads it itself.</summary>
    internal void Preload()
    {
        lock (_pairs)
            _preload ??= System.Threading.Tasks.Task.Run(() =>
            {
                try
                {
                    System.Threading.Tasks.Parallel.ForEach(_pairs.Keys.ToArray(), new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = 4 },
                        name => { if (!_disposed) Factors(name); });
                }
                catch (Exception e) when (e is ObjectDisposedException or AggregateException) { }
            });
    }

    private static bool ValidTarget(string target, int layers)
    {
        const string prefix = "transformer_blocks.";
        if (!target.StartsWith(prefix, StringComparison.Ordinal)) return false;
        int dot = target.IndexOf('.', prefix.Length);
        if (dot < 0 || !int.TryParse(target.AsSpan(prefix.Length, dot - prefix.Length), out int block) ||
            block < 0 || block >= layers || target[prefix.Length..dot] != block.ToString(System.Globalization.CultureInfo.InvariantCulture)) return false;
        return target[(dot + 1)..] is "attn1.to_q" or "attn1.to_k" or "attn1.to_v" or "attn1.to_out.0" or
            "attn2.to_q" or "attn2.to_k" or "attn2.to_v" or "attn2.to_out.0" or "ff.net.0.proj" or "ff.net.2";
    }

    /// <summary>Frees the factors' host copies; a denoiser that bound the set released their device copies first.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        try { _preload?.Wait(); }
        catch (AggregateException) { }
        lock (_loaded)
        {
            foreach (var (down, up) in _loaded.Values)
            {
                Marshal.FreeHGlobal(down.Data);
                Marshal.FreeHGlobal(up.Data);
            }
            _loaded.Clear();
        }
        _file.Dispose();
    }
}
