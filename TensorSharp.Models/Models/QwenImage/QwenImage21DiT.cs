// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using TensorSharp.Core;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models.QwenImage;

/// <summary>Qwen-Image-2.1's single-stream DiT; GGUF projections stay quantized.
/// Latent tokens are 64-channel VAE pixels, without the original Qwen-Image's 2x2 packing.</summary>
internal sealed class QwenImage21DiT : ModelBase
{
    internal const int HiddenSize = 4096, HeadDim = 128, Heads = 32, Channels = 64, Layers = 32, TextDim = 4096;
    private readonly Dictionary<string, IntPtr> _pointers = new();
    private readonly List<IntPtr> _owned = new();
    private readonly QwenImage21Block[] _blocks;
    // Per-rank block weights when the transformer is sharded over a tensor-parallel group.
    private readonly QwenImage21Block[][] _rankBlocks;
    private readonly QwenImage21ForwardArgs _nativeWeights;
    private readonly string _prefix;
    private readonly LayoutCache _layouts = new();
    // LoRA plug-ins (owned by QwenImageModel), or null.
    private readonly QwenImage21LoraSet _lora;

    /// <param name="tpGroup">When set, the blocks are sharded Megatron-style over its GPUs:
    /// whole attention heads and MLP columns per rank, two all-reduces per block.</param>
    /// <param name="lora">LoRA plug-ins applied unmerged by the native graph, or null.</param>
    public QwenImage21DiT(string ggufPath, BackendType backend, ITensorParallelGroup tpGroup = null, QwenImage21LoraSet lora = null)
        : base(ggufPath, backend, tpGroup?.Degree ?? 1, tpGroup)
    {
        _lora = lora;
        try
        {
            if (!IsGgmlBackend) throw new NotSupportedException("Qwen-Image-2.1 requires a GGML backend (ggml-metal, ggml-cuda, ggml-vulkan or ggml-cpu).");
            EnsureQuantBackendAvailable();
            Config = new ModelConfig { Architecture = "qwen_image_2_1", HiddenSize = HiddenSize, NumLayers = Layers };
            _prefix = _gguf.Tensors.ContainsKey("img_in.weight") ? "" : "model.diffusion_model.";
            _nativeWeights = new QwenImage21ForwardArgs
            {
                ImageIn = Weight("img_in.weight", Channels, HiddenSize),
                TextIn = Weight("txt_in.in_layer.weight", TextDim, HiddenSize),
                TextOut = Weight("txt_in.out_layer.weight", HiddenSize, HiddenSize),
                TimeIn = Weight("time_text_embed.timestep_embedder.linear_1.weight", 256, HiddenSize),
                TimeOut = Weight("time_text_embed.timestep_embedder.linear_2.weight", HiddenSize, HiddenSize),
                Modulation = Weight("modulation.1.weight", HiddenSize, 4 * HiddenSize),
                NormOut = Weight("norm_out.linear.weight", HiddenSize, HiddenSize),
                ProjOut = Weight("proj_out.weight", HiddenSize, Channels),
                TextNorm = lora?.TextNorm is { } norm && norm != IntPtr.Zero ? norm : F32("txt_in.text_norm.weight", TextDim),
                StructBytes = Marshal.SizeOf<QwenImage21ForwardArgs>(),
                Dim = HiddenSize, Heads = Heads, HeadDim = HeadDim, Channels = Channels, TextDim = TextDim,
                NumLayers = Layers, Eps = 1e-6f,
            };
            _blocks = new QwenImage21Block[Layers];
            for (int i = 0; i < Layers; ++i)
            {
                string p = $"transformer_blocks.{i}.";
                bool fused = _gguf.Tensors.ContainsKey(_prefix + p + "img_mlp.gate_up.weight");
                _blocks[i] = new QwenImage21Block
                {
                    Q = Weight(p + "attn.to_q.weight", HiddenSize, HiddenSize),
                    K = Weight(p + "attn.to_k.weight", HiddenSize, HiddenSize),
                    V = Weight(p + "attn.to_v.weight", HiddenSize, HiddenSize),
                    Out = Weight(p + "attn.to_out.0.weight", HiddenSize, HiddenSize),
                    Gate = Weight(p + (fused ? "img_mlp.gate_up.weight" : "img_mlp.gate_layer.weight"), HiddenSize, fused ? 24576 : 12288),
                    Up = fused ? default : Weight(p + "img_mlp.proj.weight", HiddenSize, 12288),
                    Down = Weight(p + "img_mlp.out.weight", 12288, HiddenSize),
                    NormQ = lora != null && lora.NormQ[i] != IntPtr.Zero ? lora.NormQ[i] : F32(p + "attn.norm_q.weight", HeadDim),
                    NormK = lora != null && lora.NormK[i] != IntPtr.Zero ? lora.NormK[i] : F32(p + "attn.norm_k.weight", HeadDim),
                };
                // A LoRA on the gate or up half of a fused projection: describe the two
                // halves as row views so each takes its own update before SwiGLU.
                if (fused && lora != null && lora.SplitGateUp[i])
                {
                    var gateUp = _blocks[i].Gate;
                    _blocks[i].Gate = Rows(gateUp, 0, 12288);
                    _blocks[i].Up = Rows(gateUp, 12288, 12288);
                }
            }
            if (_gguf.Tensors.ContainsKey(_prefix + $"transformer_blocks.{Layers}.attn.to_q.weight"))
                throw new NotSupportedException("Expected a 32-layer Qwen-Image-2.1 transformer.");
            if (IsTensorParallel)
            {
                _rankBlocks = ShardBlocks(_blocks, TpDegree, _owned);
                Console.WriteLine($"Qwen-Image-2.1 DiT: {Layers} layers, {HiddenSize} hidden, {Heads} heads sharded over {TpDegree} GPUs " +
                    $"({Heads / TpDegree} heads and {12288 / TpDegree} MLP columns each), quantized resident GGML graphs.");
            }
            else
                Console.WriteLine($"Qwen-Image-2.1 DiT: {Layers} layers, {HiddenSize} hidden, {Heads} heads, quantized resident GGML graph.");
        }
        catch { Dispose(); throw; }
    }

    /// <summary>Megatron sharding of the blocks. Q/K/V, gate and up keep rows (outputs) per
    /// rank, contiguous in every GGML type, so they are views; to_out and img_mlp.out keep
    /// columns (inputs), which are copied block-aligned per row. A fused [gate; up]
    /// projection becomes separate gate and up slices so each rank's pair stays matched.</summary>
    internal static QwenImage21Block[][] ShardBlocks(QwenImage21Block[] blocks, int ranks, List<IntPtr> owned)
    {
        if (ranks < 2 || Heads % ranks != 0)
            throw new NotSupportedException($"Qwen-Image-2.1 tensor parallelism needs a GPU count that divides its {Heads} heads, not {ranks}.");
        var result = new QwenImage21Block[ranks][];
        for (int r = 0; r < ranks; r++)
        {
            result[r] = new QwenImage21Block[blocks.Length];
            for (int i = 0; i < blocks.Length; i++)
            {
                var b = blocks[i];
                long ff = b.Down.Ne0, ffLocal = ff / ranks, local = b.Q.Ne1 / ranks;
                result[r][i] = new QwenImage21Block
                {
                    Q = Rows(b.Q, r * local, local),
                    K = Rows(b.K, r * local, local),
                    V = Rows(b.V, r * local, local),
                    Out = Columns(b.Out, r * local, local, ranks, owned),
                    Gate = Rows(b.Gate, r * ffLocal, ffLocal),
                    Up = b.Up.Data != IntPtr.Zero ? Rows(b.Up, r * ffLocal, ffLocal) : Rows(b.Gate, ff + r * ffLocal, ffLocal),
                    Down = Columns(b.Down, r * ffLocal, ffLocal, ranks, owned),
                    NormQ = b.NormQ,
                    NormK = b.NormK,
                };
            }
        }
        return result;
    }

    private static QwenImage21Weight Rows(QwenImage21Weight w, long start, long count)
    {
        long rowBytes = w.Bytes / w.Ne1;
        return w with { Data = w.Data + checked((nint)(start * rowBytes)), Ne1 = count, Bytes = count * rowBytes };
    }

    private static unsafe QwenImage21Weight Columns(QwenImage21Weight w, long start, long count, int ranks, List<IntPtr> owned)
    {
        var type = (GgmlTensorType)w.Type;
        long block = GgufFile.GetBlockSize(type), typeSize = GgufFile.GetTypeSize(type);
        if (w.Ne0 % (block * ranks) != 0)
            throw new NotSupportedException(
                $"Qwen-Image-2.1 tensor parallelism cannot split a {w.Ne0}-wide {type} row over {ranks} GPUs on {block}-element blocks.");
        long sourceRow = w.Bytes / w.Ne1, row = count / block * typeSize, offset = start / block * typeSize;
        IntPtr data = QuantizedWeight.AllocateBuffer(checked(row * w.Ne1));
        owned.Add(data);
        byte* source = (byte*)w.Data, destination = (byte*)data;
        for (long o = 0; o < w.Ne1; o++)
            Buffer.MemoryCopy(source + o * sourceRow + offset, destination + o * row, row, row);
        return w with { Data = data, Ne0 = count, Bytes = row * w.Ne1 };
    }

    private QwenImage21Weight Weight(string name, int input, int output)
    {
        name = _prefix + name;
        if (!_gguf.Tensors.TryGetValue(name, out var info) || info.Shape.Length != 2 ||
            (long)info.Shape[0] != input || (long)info.Shape[1] != output)
            throw new NotSupportedException($"Qwen-Image-2.1 requires {name} with GGUF shape [{input},{output}].");
        if (!_gguf.TryGetTensorDataPointer(info, out IntPtr ptr))
        {
            ptr = QuantizedWeight.AllocateBuffer(_gguf.GetTensorByteCount(info));
            _owned.Add(ptr);
            _gguf.ReadTensorDataToNative(info, ptr, _gguf.GetTensorByteCount(info));
        }
        return new QwenImage21Weight { Data = ptr, Type = (int)info.Type, Ne0 = input, Ne1 = output, Bytes = _gguf.GetTensorByteCount(info) };
    }

    private IntPtr F32(string name, int count)
    {
        name = _prefix + name;
        if (_pointers.TryGetValue(name, out var ptr)) return ptr;
        if (!_gguf.Tensors.TryGetValue(name, out var info) || info.NumElements != count)
            throw new NotSupportedException($"Qwen-Image-2.1 requires {name} with {count} elements.");
        if (info.Type == GgmlTensorType.F32 && _gguf.TryGetTensorDataPointer(info, out ptr))
            return _pointers[name] = ptr;
        ptr = QuantizedWeight.AllocateBuffer(count * sizeof(float));
        _owned.Add(ptr);
        long bytes = _gguf.GetTensorByteCount(info);
        IntPtr source = QuantizedWeight.AllocateBuffer(bytes);
        try
        {
            _gguf.ReadTensorDataToNative(info, source, bytes);
            NativeDequant.DequantizeToFloat32Native((int)info.Type, source, ptr, count);
        }
        finally { QuantizedWeight.FreeBuffer(source); }
        return _pointers[name] = ptr;
    }

    /// <summary>GPUs the blocks are sharded over (1 when unsharded).</summary>
    internal int TensorParallelRanks => _rankBlocks?.Length ?? 1;

    private static long s_prefixKeys;

    /// <summary>One request's prefix KV cache: the per-layer K/V of its text and reference
    /// tokens, which are modulated at t=0 and so do not change between denoising steps.
    /// The first prediction stores them; later ones compute only the target tokens.
    /// It is bound to the exact conditioning arrays it was created for.</summary>
    internal sealed class PrefixCache : IDisposable
    {
        internal PrefixCache(float[] text, int[] slots, float[][] references, QwenImage21PrefixCacheType type)
        {
            Key = (ulong)Interlocked.Increment(ref s_prefixKeys);
            Text = text; Slots = slots; References = references; Type = type;
        }

        internal ulong Key { get; }
        internal float[] Text { get; }
        internal int[] Slots { get; }
        internal float[][] References { get; }
        internal QwenImage21PrefixCacheType Type { get; }
        /// <summary>The graph that produced the most recent prediction.</summary>
        internal QwenImage21ForwardPath LastPath { get; set; }
        private bool _disposed;

        internal QwenImage21PrefixCacheInfo Info => GgmlBasicOps.QwenImage21GetPrefixCacheInfo(Key);

        /// <summary>True when a prediction passes the very arrays this cache was made for.
        /// Stored K/V encode that conditioning; other text or references would silently
        /// condition on the wrong prompt, so identity rather than content is required.</summary>
        internal bool Describes(float[] text, int[] slots, float[][] references) =>
            ReferenceEquals(Text, text) && ReferenceEquals(Slots, slots) &&
            References.AsSpan().SequenceEqual(references ?? Array.Empty<float[]>());

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            GgmlBasicOps.QwenImage21ReleasePrefixCache(Key);
        }
    }

    /// <summary>Creates a request's prefix cache, or null when TS_QWEN21_PREFIX_CACHE=0.
    /// TS_QWEN21_PREFIX_CACHE_TYPE selects auto (default), f32, f16, q8_0 or q8_0_v.</summary>
    internal static PrefixCache CreatePrefixCache(float[] textCond, int[] imageSlots, float[][] referenceTokens) =>
        CreatePrefixCache(textCond, imageSlots, referenceTokens,
            Environment.GetEnvironmentVariable("TS_QWEN21_PREFIX_CACHE"),
            Environment.GetEnvironmentVariable("TS_QWEN21_PREFIX_CACHE_TYPE"));

    internal static PrefixCache CreatePrefixCache(float[] textCond, int[] imageSlots, float[][] referenceTokens,
        string enabled, string type)
    {
        ArgumentNullException.ThrowIfNull(textCond);
        // Parse first: a misspelled type must fail even while the cache is disabled.
        var storage = ParsePrefixCacheType(type);
        if (enabled != null && enabled.Trim().ToLowerInvariant() is "0" or "false" or "off" or "no") return null;
        return new PrefixCache(textCond, imageSlots, referenceTokens ?? Array.Empty<float[]>(), storage);
    }

    internal static QwenImage21PrefixCacheType ParsePrefixCacheType(string value) =>
        (value ?? "").Trim().ToLowerInvariant() switch
        {
            "" or "auto" => QwenImage21PrefixCacheType.Auto,
            "f32" => QwenImage21PrefixCacheType.F32,
            "f16" => QwenImage21PrefixCacheType.F16,
            "q8_0" => QwenImage21PrefixCacheType.Q8_0,
            "q8_0_v" => QwenImage21PrefixCacheType.Q8_0V,
            _ => throw new ArgumentException(
                $"TS_QWEN21_PREFIX_CACHE_TYPE='{value}' is not one of auto, f32, f16, q8_0, q8_0_v."),
        };

    /// <param name="imageSlots">One tag per text token: 0=text, 1..N=reference image.
    /// Each contiguous vision-slot run is replaced by four times as many latent tokens.</param>
    /// <param name="prefixCache">Optional cache created for these same conditioning arrays.</param>
    /// <param name="step">Denoising step index; selects a LoRA bundle's per-step output head.</param>
    /// <param name="bf16Timestep">Round the timestep as a bf16 pipeline does: sigma*1000 to bf16, then /1000 in bf16.</param>
    internal float[] Predict(float[] targetTokens, int latentH, int latentW, float[] textCond, int textSeq,
        float timestep01, int[] imageSlots = null, float[][] referenceTokens = null,
        int[] referenceHeights = null, int[] referenceWidths = null, PrefixCache prefixCache = null,
        int step = 0, bool bf16Timestep = false)
    {
        if (latentH <= 0 || latentW <= 0 || targetTokens == null || targetTokens.Length != checked(latentH * latentW * Channels))
            throw new ArgumentException("Target must contain latentH*latentW*64 token-major floats.");
        if (textSeq <= 0 || textCond == null || textCond.Length != checked(textSeq * TextDim))
            throw new ArgumentException("Conditioning must contain textSeq*4096 token-major floats.");
        if (!float.IsFinite(timestep01) || timestep01 < 0 || timestep01 > 1)
            throw new ArgumentOutOfRangeException(nameof(timestep01));
        referenceTokens ??= Array.Empty<float[]>();
        referenceHeights ??= Array.Empty<int>();
        referenceWidths ??= Array.Empty<int>();
        if (referenceHeights.Length != referenceTokens.Length || referenceWidths.Length != referenceTokens.Length)
            throw new ArgumentException("Reference latent shapes are required for each reference image.");
        for (int i = 0; i < referenceTokens.Length; ++i)
            if (referenceHeights[i] <= 0 || referenceWidths[i] <= 0 || referenceTokens[i] == null ||
                referenceTokens[i].Length != checked(referenceHeights[i] * referenceWidths[i] * Channels))
                throw new ArgumentException($"Invalid reference latent {i}.");
        // A cache stores K/V computed from its conditioning; reusing it for other
        // text or references would silently condition on the wrong prompt.
        if (prefixCache != null && !prefixCache.Describes(textCond, imageSlots, referenceTokens))
            throw new ArgumentException("The prefix cache belongs to different conditioning.", nameof(prefixCache));
        var shapes = referenceHeights.Select((h, i) => (Height: h, Width: referenceWidths[i])).Append((latentH, latentW)).ToArray();
        var layout = _layouts.Get(textSeq, imageSlots, shapes);
        // Text-to-image already has the native token layout. Pin the caller's
        // latents directly instead of copying them on every denoising step.
        float[] images = targetTokens;
        if (referenceTokens.Length != 0)
        {
            images = new float[checked(referenceTokens.Sum(x => x.Length) + targetTokens.Length)];
            int offset = 0;
            foreach (var reference in referenceTokens) { reference.CopyTo(images, offset); offset += reference.Length; }
            targetTokens.CopyTo(images, offset);
        }
        if (bf16Timestep) timestep01 = RoundBf16(RoundBf16(timestep01 * 1000f) / 1000f);
        _lora?.SelectOutputHead(step);
        var time = new float[512];
        for (int i = 0; i < 128; ++i)
        {
            float angle = timestep01 * 1000f * MathF.Exp(-MathF.Log(10000f) * i / 128);
            time[i] = MathF.Cos(angle); time[i + 128] = MathF.Sin(angle);
            time[256 + i] = 1f;
        }
        var output = new float[targetTokens.Length];
        var pins = new List<GCHandle>();
        IntPtr Pin<T>(T[] data) where T : struct
        {
            var handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            pins.Add(handle);
            return handle.AddrOfPinnedObject();
        }
        try
        {
            var args = _nativeWeights;
            args.Images = Pin(images); args.Text = Pin(textCond); args.TimeEmbedding = Pin(time);
            args.Cos = Pin(layout.Cos); args.Sin = Pin(layout.Sin); args.Output = Pin(output);
            args.Blocks = Pin(_blocks); args.Segments = Pin(layout.Segments);
            args.ImageSeq = images.Length / Channels; args.TextSeq = textSeq;
            args.TotalSeq = layout.Cos.Length / (HeadDim / 2); args.PrefixSeq = layout.Prefix;
            args.NumSegments = layout.Segments.Length;
            args.PrefixCacheKey = prefixCache?.Key ?? 0;
            args.PrefixCacheType = prefixCache?.Type ?? QwenImage21PrefixCacheType.Auto;
            args.Adapter = _lora?.AdapterFor(0) ?? IntPtr.Zero;
            QwenImage21ForwardPath path;
            if (_rankBlocks == null)
                path = GgmlBasicOps.QwenImage21Forward(in args);
            else
            {
                var ranks = new QwenImage21ForwardArgs[_rankBlocks.Length];
                for (int r = 0; r < ranks.Length; r++)
                {
                    ranks[r] = args;
                    ranks[r].Blocks = Pin(_rankBlocks[r]);
                    ranks[r].Heads = Heads / ranks.Length;
                    ranks[r].TpRanks = ranks.Length;
                    ranks[r].Adapter = _lora?.AdapterFor(r) ?? IntPtr.Zero;
                }
                path = GgmlBasicOps.QwenImage21ForwardTp(ranks);
            }
            if (prefixCache != null) prefixCache.LastPath = path;
        }
        finally { foreach (var pin in pins) pin.Free(); }
        if (output.Any(v => !float.IsFinite(v)))
            throw new InvalidOperationException("Qwen-Image-2.1 DiT produced non-finite latent velocities.");
        return output;
    }

    /// <summary>Round to the nearest bfloat16 (ties to even), as torch's .to(torch.bfloat16).</summary>
    internal static float RoundBf16(float value)
    {
        int bits = BitConverter.SingleToInt32Bits(value);
        bits += 0x7FFF + ((bits >> 16) & 1);
        return BitConverter.Int32BitsToSingle(bits & unchecked((int)0xFFFF0000));
    }

    /// <summary>Retains at most the two CFG layouts. Keys are copied because callers
    /// may reuse and mutate slot/shape arrays between image requests.</summary>
    internal sealed class LayoutCache
    {
        private sealed record Entry(int TextLength, int[] Slots, (int Height, int Width)[] Shapes,
            (QwenImage21Segment[] Segments, float[] Cos, float[] Sin, int Prefix) Layout);
        private Entry _recent, _previous;

        internal (QwenImage21Segment[] Segments, float[] Cos, float[] Sin, int Prefix) Get(
            int textLength, int[] slots, (int Height, int Width)[] shapes)
        {
            bool Matches(Entry entry) => entry != null && entry.TextLength == textLength &&
                ((slots == null && entry.Slots == null) ||
                 (slots != null && entry.Slots != null && slots.AsSpan().SequenceEqual(entry.Slots))) &&
                shapes != null && shapes.AsSpan().SequenceEqual(entry.Shapes);
            if (Matches(_recent)) return _recent.Layout;
            if (Matches(_previous))
            {
                (_recent, _previous) = (_previous, _recent);
                return _recent.Layout;
            }
            var layout = BuildLayout(textLength, slots, shapes);
            _previous = _recent;
            _recent = new Entry(textLength, slots?.ToArray(), shapes.ToArray(), layout);
            return layout;
        }

        internal void Clear() => (_recent, _previous) = (null, null);
    }

    internal static (QwenImage21Segment[] Segments, float[] Cos, float[] Sin, int Prefix) BuildLayout(
        int textLength, int[] imageSlots, (int Height, int Width)[] shapes)
    {
        if (textLength <= 0 || shapes == null || shapes.Length == 0 ||
            shapes.Any(s => s.Height <= 0 || s.Width <= 0) || (imageSlots != null && imageSlots.Length != textLength))
            throw new ArgumentException("Invalid Qwen-Image-2.1 token layout.");
        var segments = new List<QwenImage21Segment>();
        var positions = new List<(int T, int H, int W)>();
        int position = 0, nextImage = 0, imageOffset = 0;
        void AppendImage(int index)
        {
            var (height, width) = shapes[index];
            int count = checked(height * width), start = positions.Count;
            segments.Add(new QwenImage21Segment { Start = start, End = checked(start + count), SourceStart = imageOffset, IsImage = 1 });
            for (int h = 0; h < height; ++h)
                for (int w = 0; w < width; ++w)
                    positions.Add((position, h - (height - height / 2), w - (width - width / 2)));
            position += Math.Max(height, width); imageOffset += count;
        }
        for (int i = 0; i < textLength;)
        {
            int begin = i, tag = imageSlots?[i] ?? 0;
            while (i < textLength && (imageSlots?[i] ?? 0) == tag) ++i;
            if (tag != 0)
            {
                if (tag != nextImage + 1 || nextImage + 1 >= shapes.Length ||
                    (long)(i - begin) * 4 != (long)shapes[nextImage].Height * shapes[nextImage].Width)
                    throw new ArgumentException("Vision slots and reference latents must have matching sizes and ordered image tags.");
                AppendImage(nextImage++);
            }
            else
            {
                int start = positions.Count;
                segments.Add(new QwenImage21Segment { Start = start, End = start + i - begin, SourceStart = begin });
                for (int j = begin; j < i; ++j, ++position) positions.Add((position, position, position));
            }
        }
        if (nextImage + 1 != shapes.Length) throw new ArgumentException("Missing reference image slots.");
        int prefix = positions.Count;
        AppendImage(nextImage);
        var cos = new float[checked(positions.Count * HeadDim / 2)];
        var sin = new float[cos.Length];
        int[] dims = { 16, 56, 56 };
        for (int token = 0; token < positions.Count; ++token)
        {
            var p = positions[token];
            int[] coords = { p.T, p.H, p.W };
            int channel = 0;
            for (int axis = 0; axis < 3; ++axis)
                for (int j = 0; j < dims[axis] / 2; ++j, ++channel)
                {
                    double angle = coords[axis] / Math.Pow(10000.0, 2.0 * j / dims[axis]);
                    cos[token * HeadDim / 2 + channel] = (float)Math.Cos(angle);
                    sin[token * HeadDim / 2 + channel] = (float)Math.Sin(angle);
                }
        }
        return (segments.ToArray(), cos, sin, prefix);
    }

    // The group belongs to the QwenImageModel that created this transformer.
    protected override bool OwnsTensorParallelGroup => false;
    protected override float[] ForwardCore(int[] tokens) => throw new NotSupportedException("Use Predict for image inference.");
    protected override void ResetKVCacheCore() { }
    public override void Dispose()
    {
        _layouts.Clear();
        base.Dispose();
        foreach (var ptr in _owned) QuantizedWeight.FreeBuffer(ptr);
        _owned.Clear(); _pointers.Clear();
    }
}
