// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.Direct;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Ltx25;

/// <summary>The publisher's default LTX 2.5 video decoder (<c>CausalDiffusionVAE</c> /
/// <c>NADiffusionDecoder</c>, <c>ltx-2.5-video-vae-bf16.safetensors</c>). Four
/// deterministic neighborhood-attention stages upsample the latent into a context
/// volume; a fifth, diffusion stage predicts the pixels (x0) in one step from
/// Gaussian pixel noise at t = 1, guided by that context through AdaLN modulation.
/// It is generative: it renders detail the convolutional decoder averages away.
///
/// The decode follows the original untiled <c>_decode_pixels</c>: latents below
/// the windows' floor are edge padded, the last latent frame is repeated twice for
/// NATTEN's trailing border and cropped again before the diffusion stage, and the
/// output is cropped to the content. Every stage keeps one tensor per frame;
/// attention runs over temporal chunks against cached pre-update Q/K/V frames, so
/// no full-volume attention operand exists. BF16 weights widen exactly and are
/// streamed per block; activations are Float32 on the CPU or MLX allocator.</summary>
public sealed class Ltx25DiffusionVideoDecoder : IDisposable
{
    public const int TemporalScale = 8, SpatialScale = 32;
    /// <summary>The Philox stream the decoder's pixel noise is drawn from, apart
    /// from the first pass (0) and the refinement pass (1).</summary>
    public const ulong NoiseStream = 2;
    private const long ChunkBudgetBytes = 512L << 20;
    private const int AdaLnChunks = 7;
    private readonly SafetensorsFile _file;
    private readonly DirectContext _context;
    private readonly Ltx25DiffusionVaeConfig _config;
    private readonly object _gate = new();
    private readonly float[] _ropeFrequencies;
    private bool _disposed;
    public int LatentChannels => _config.LatentChannels;

    public Ltx25DiffusionVideoDecoder(string path, IAllocator allocator)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(allocator);
        if (allocator is CudaAllocator)
            throw new NotSupportedException("The LTX 2.5 diffusion video decoder runs on MLX and CPU. On CUDA, decode with the " +
                "convolutional VAE: --video-vae ltx-2.5-video-vae-conv-bf16.safetensors (TS_VIDEO_VAE).");
        if (allocator is not (CpuAllocator or MlxAllocator))
            throw new NotSupportedException("The LTX diffusion video decoder requires a CPU or MLX direct allocator.");
        _context = new DirectContext(allocator);
        _file = new SafetensorsFile(path);
        try
        {
            _config = Ltx25DiffusionVaeConfig.Parse(_file.Metadata);
            ValidateWeights();
            _ropeFrequencies = Ltx25DiffusionVaeKernels.RopeFrequencies(_config.RopeSplit, Ltx25DiffusionVaeConfig.RopeBase);
        }
        catch { _file.Dispose(); _context.Dispose(); throw; }
    }

    private static int Hidden(int channels) => (channels * 4 + 15) / 16 * 16;

    private void ValidateWeights()
    {
        void Shape(string name, params long[] expected)
        {
            if (!_file.HasTensor(name) || !_file.TensorShape(name).SequenceEqual(expected))
                throw new InvalidDataException($"LTX diffusion VAE tensor '{name}' must have shape [{string.Join(',', expected)}].");
            if (_file.GetInfo(name).Dtype is not (SafetensorDtype.BF16 or SafetensorDtype.F16 or SafetensorDtype.F32))
                throw new NotSupportedException($"LTX diffusion VAE tensor '{name}' must use BF16, F16 or Float32 storage.");
        }
        void Linear(string name, int input, int output, bool bias = true)
        { Shape("decoder." + name + ".weight", output, input); if (bias) Shape("decoder." + name + ".bias", output); }
        void Block(string name, int channels)
        {
            Shape("decoder." + name + ".norm1.weight", channels);
            Linear(name + ".attn.qkv", channels, 3 * channels);
            Shape("decoder." + name + ".attn.q_norm.weight", _config.HeadDim);
            Shape("decoder." + name + ".attn.k_norm.weight", _config.HeadDim);
            Linear(name + ".attn.proj", channels, channels);
            Shape("decoder." + name + ".norm2.weight", channels);
            Linear(name + ".mlp.w_gate", channels, Hidden(channels), false);
            Linear(name + ".mlp.w_up", channels, Hidden(channels), false);
            Linear(name + ".mlp.w_down", Hidden(channels), channels, false);
        }
        var c = _config.StageChannels;
        Shape("per_channel_statistics.mean-of-means", LatentChannels);
        Shape("per_channel_statistics.std-of-means", LatentChannels);
        Linear("conv_in", LatentChannels, c[0]);
        for (int stage = 0; stage < 4; ++stage)
        {
            for (int block = 0; block < _config.StageDepths[stage]; ++block) Block($"det_stages.{stage}.{block}", c[stage]);
            var upsample = _config.Upsamples[stage];
            Linear($"upsamples.{stage}.proj", c[stage], c[stage] / upsample.Reduction * upsample.Stride.Aggregate(1, (a, b) => a * b));
        }
        int temb = _config.TimestepEmbedDim, c5 = _config.Stage5Channels;
        Linear("t_embedder.mlp.0", 256, temb);
        Linear("t_embedder.mlp.2", temb, temb);
        Linear("shared_adaln.proj", temb, AdaLnChunks * c5);
        Linear("conv_in_x_t", _config.PixelChannels, c5);
        for (int block = 0; block < _config.DiffusionBlocks; ++block)
        {
            Block($"diff_blocks.{block}", c5);
            Linear($"diff_blocks.{block}.context_proj", _config.ContextChannels, c5);
            Shape($"decoder.diff_blocks.{block}.scale_shift_table", AdaLnChunks, c5);
        }
        Shape("decoder.norm_out.weight", c5);
        Linear("conv_out", c5, _config.PixelChannels);
    }

    /// <summary>Stage-five pixel canvas [3,F,H,W] for content latents [C,T,H,W]: what
    /// <see cref="DecodeRaw"/> expects as noise. It covers the floor padding, and at
    /// least the diffusion window's frames.</summary>
    public int[] NoiseShape(int frames, int height, int width)
    {
        var geometry = Geometry(frames, height, width);
        return [3, geometry.CanvasFrames, geometry.CanvasHeight, geometry.CanvasWidth];
    }

    /// <summary>Decode channel-major [C,T,H,W] normalized latents to frame-major
    /// [8*(T-1)+1, 32*H, 32*W, 3] RGB in [0,1]. Pixel noise is drawn per canvas frame
    /// from <paramref name="seed"/> on <see cref="NoiseStream"/>.</summary>
    public float[] Decode(float[] latents, int frames, int height, int width, long seed, CancellationToken cancellationToken = default)
    {
        var geometry = Geometry(frames, height, width);
        int pixels = geometry.CanvasHeight * geometry.CanvasWidth;
        int frameSize = checked(height * SpatialScale * width * SpatialScale * 3);
        var result = new float[checked((long)geometry.ContentFrames * frameSize)];
        Run(latents, frames, height, width, frame => PhiloxNoise.Draw(checked(3 * pixels), seed, NoiseStream, (uint)frame),
            (frame, rgb) =>
            {
                // Planar [3, H, W] in [-1, 1] -> interleaved RGB in [0, 1].
                int area = height * SpatialScale * width * SpatialScale;
                long baseOffset = (long)frame * frameSize;
                for (int i = 0; i < area; ++i)
                    for (int channel = 0; channel < 3; ++channel)
                    {
                        float value = rgb[channel * area + i];
                        if (!float.IsFinite(value)) throw new InvalidOperationException("LTX diffusion decoder produced a non-finite pixel.");
                        result[baseOffset + i * 3 + channel] = Math.Clamp((value + 1f) * .5f, 0f, 1f);
                    }
            }, cancellationToken, null);
        return result;
    }

    /// <summary>Decode to unclamped channel-major [3,8*(T-1)+1,32*H,32*W], the
    /// original <c>_decode_pixels</c> layout, from caller-supplied noise of
    /// <see cref="NoiseShape"/>. Optional traces receive channels-last stage tensors.</summary>
    public float[] DecodeRaw(float[] latents, int frames, int height, int width, float[] noise,
        CancellationToken cancellationToken = default, Action<string, float[], int[]>? trace = null)
    {
        ArgumentNullException.ThrowIfNull(noise);
        var geometry = Geometry(frames, height, width);
        int canvas = geometry.CanvasHeight * geometry.CanvasWidth;
        if (noise.Length != checked(3L * geometry.CanvasFrames * canvas))
            throw new ArgumentException("Noise must cover the decoder's [3,F,H,W] pixel canvas.", nameof(noise));
        int area = height * SpatialScale * width * SpatialScale;
        var result = new float[checked(3L * geometry.ContentFrames * area)];
        Run(latents, frames, height, width, frame =>
            {
                var slice = new float[3 * canvas];
                for (int channel = 0; channel < 3; ++channel)
                    Array.Copy(noise, ((long)channel * geometry.CanvasFrames + frame) * canvas, slice, channel * canvas, canvas);
                return slice;
            },
            (frame, rgb) =>
            {
                for (int channel = 0; channel < 3; ++channel)
                    Array.Copy(rgb, channel * area, result, ((long)channel * geometry.ContentFrames + frame) * area, area);
            }, cancellationToken, trace);
        return result;
    }

    private sealed record DecodeGeometry(int[] Padded, int[] Before, int ContentFrames, int ContentHeight, int ContentWidth,
        int CanvasFrames, int CanvasHeight, int CanvasWidth, int Ghost);

    private DecodeGeometry Geometry(int frames, int height, int width)
    {
        if (frames <= 0 || height <= 0 || width <= 0) throw new ArgumentOutOfRangeException(nameof(frames), "Latent dimensions must be positive.");
        int[] minimum = _config.MinimumLatent(), content = [frames, height, width];
        int[] padded = new int[3], before = new int[3];
        for (int axis = 0; axis < 3; ++axis)
        {
            padded[axis] = Math.Max(content[axis], minimum[axis]);
            // Time repeats its last frame; rows and columns split the pad, leftover after.
            before[axis] = axis == 0 ? 0 : (padded[axis] - content[axis]) / 2;
        }
        int ghost = _config.StageKernels[0][0] / 2 * 2;
        // Stage-4 input frames of the padded latent, without the ghost frames.
        int stage4 = padded[0], stage4H = padded[1], stage4W = padded[2];
        for (int stage = 0; stage < 3; ++stage)
        {
            var stride = _config.Upsamples[stage].Stride;
            stage4 = stride[0] == 2 ? stage4 * 2 - 1 : stage4;
            stage4H *= stride[1]; stage4W *= stride[2];
        }
        var last = _config.Upsamples[3].Stride;
        int canvasFrames = Math.Max(last[0] == 2 ? stage4 * 2 - 1 : stage4, _config.Stage5Kernel[0]);
        return new(padded, before, (frames - 1) * TemporalScale + 1, height * SpatialScale, width * SpatialScale, canvasFrames,
            stage4H * last[1] * _config.PatchSize, stage4W * last[2] * _config.PatchSize, ghost);
    }

    /// <summary>Linear layer on device: y[rows, out] = x[rows, in] W^T + b.</summary>
    private sealed class Linear : IDisposable
    {
        private readonly DirectContext _context;
        private readonly Tensor _weight, _bias;
        public int Output { get; }
        public Linear(DirectContext context, float[] weight, float[] bias, int input, int output)
        {
            _context = context; Output = output;
            _weight = context.FromFloats(weight, output, input);
            try { _bias = bias == null ? null : context.FromFloats(bias, output); }
            catch { _weight.Dispose(); throw; }
        }
        public Tensor Forward(Tensor x)
        {
            var y = _context.NewF32(x.Sizes[0], Output);
            try
            {
                DirectOps.CpuGemmABt(x, _weight, y, 1f, 0f);
                if (_bias != null) DirectOps.AddBiasRows(_context, y, _bias);
                return y;
            }
            catch { y.Dispose(); throw; }
        }
        public void Dispose() { _weight.Dispose(); _bias?.Dispose(); }
    }

    private float[] Read(string name) => _file.ReadFloat32(name);

    /// <summary>Reads a Linear, folding a legacy static gate (gate_msa / gate_mlp /
    /// gate_ctx) into its rows as the original loader does. The published checkpoint has none.</summary>
    private Linear LoadLinear(string name, string gate = null, int rows = 0, int rowOffset = 0)
    {
        string key = "decoder." + name;
        var shape = _file.TensorShape(key + ".weight");
        int output = checked((int)shape[0]), input = checked((int)shape[1]);
        float[] weight = Read(key + ".weight");
        float[] bias = _file.HasTensor(key + ".bias") ? Read(key + ".bias") : null;
        if (rows > 0)
        {
            weight = weight.AsSpan(rowOffset * input, rows * input).ToArray();
            bias = bias?.AsSpan(rowOffset, rows).ToArray();
            output = rows;
        }
        if (gate != null && _file.HasTensor("decoder." + gate))
        {
            float[] g = Read("decoder." + gate);
            if (g.Length != output) throw new InvalidDataException($"LTX diffusion VAE gate '{gate}' does not match '{name}'.");
            for (int row = 0; row < output; ++row)
            {
                for (int column = 0; column < input; ++column) weight[row * input + column] *= g[row];
                if (bias != null) bias[row] *= g[row];
            }
        }
        return new Linear(_context, weight, bias, input, output);
    }

    /// <summary>One transformer block's attention weights.</summary>
    private sealed class AttentionWeights : IDisposable
    {
        public Linear Q, K, V, Projection; public Tensor QNorm, KNorm;
        public void Dispose() { Q?.Dispose(); K?.Dispose(); V?.Dispose(); Projection?.Dispose(); QNorm?.Dispose(); KNorm?.Dispose(); }
    }

    private AttentionWeights LoadAttention(string block, int channels, bool diffusion)
    {
        var weights = new AttentionWeights();
        try
        {
            weights.Q = LoadLinear(block + ".attn.qkv", rows: channels, rowOffset: 0);
            weights.K = LoadLinear(block + ".attn.qkv", rows: channels, rowOffset: channels);
            weights.V = LoadLinear(block + ".attn.qkv", rows: channels, rowOffset: 2 * channels);
            weights.Projection = LoadLinear(block + ".attn.proj", diffusion ? block + ".gate_msa" : null);
            weights.QNorm = _context.FromFloats(Read("decoder." + block + ".attn.q_norm.weight"), _config.HeadDim);
            weights.KNorm = _context.FromFloats(Read("decoder." + block + ".attn.k_norm.weight"), _config.HeadDim);
            return weights;
        }
        catch { weights.Dispose(); throw; }
    }

    private static int ChunkFrames(int frames, int height, int width, int channels)
        => (int)Math.Clamp(ChunkBudgetBytes / Math.Max(1L, (long)height * width * channels * sizeof(float)), 1, frames);

    private void Evaluate(Tensor tensor)
    {
        if (_context.IsMlx) MlxLtx25VaeOps.Materialize(tensor);
    }

    /// <summary>Concatenate per-frame [H*W, C] tensors into [n*H*W, C].</summary>
    private Tensor Concatenate(IReadOnlyList<Tensor> frames)
    {
        long rows = frames[0].Sizes[0], channels = frames[0].Sizes[1];
        var output = _context.NewF32(rows * frames.Count, channels);
        try
        {
            if (_context.IsMlx)
            {
                using var target = output.View(frames.Count, rows, 1, channels);
                var views = frames.Select(frame => frame.View(1, rows, 1, channels)).ToArray();
                try
                {
                    if (!MlxLtx25VaeOps.TryConcatenateFrames(target, views))
                        throw new NotSupportedException("MLX LTX frame concatenation is unavailable.");
                }
                finally { foreach (var view in views) view.Dispose(); }
                MlxLtx25VaeOps.Materialize(output);
                return output;
            }
            for (int i = 0; i < frames.Count; ++i)
            {
                using var slot = output.Narrow(0, i * rows, rows);
                Ops.Copy(slot, frames[i]);
            }
            return output;
        }
        catch { output.Dispose(); throw; }
    }

    private static Tensor[] Split(Tensor chunk, int count)
    {
        long rows = chunk.Sizes[0] / count;
        var frames = new Tensor[count];
        for (int i = 0; i < count; ++i) frames[i] = chunk.Narrow(0, i * rows, rows);
        return frames;
    }

    private Tensor RmsNorm(Tensor x, Tensor gain) => DirectOps.RmsNorm(_context, x, gain, Ltx25DiffusionVaeConfig.NormEpsilon);

    /// <summary>x += proj(NA(q, k, v)) over the whole [frames, H, W] volume, where
    /// q/k/v come from <paramref name="prenorm"/>(x) evaluated before this residual.
    /// Each chunk's keys come from cached per-frame projections computed before any
    /// of their frames were updated; a frame's projections are computed when it
    /// first enters a window, which is at or before its own chunk. A positive
    /// <paramref name="padW"/> gives keys and values edge-replicated columns with
    /// continuing rotary positions (the original W-slab pathway's volume edges).</summary>
    private void AttentionResidual(List<Tensor> x, int height, int width, int channels, int[] kernel,
        Func<Tensor, Tensor> prenorm, AttentionWeights weights, CancellationToken cancellationToken, int padW = 0)
    {
        int frames = x.Count, heads = channels / _config.HeadDim, headDim = _config.HeadDim, kt = Math.Min(kernel[0], frames);
        int chunk = ChunkFrames(frames, height, width, channels);
        float scale = 1f / MathF.Sqrt(headDim);
        var split = _config.RopeSplit;
        var q = new Tensor[frames]; var k = new Tensor[frames]; var v = new Tensor[frames];
        int computed = 0;
        using var frequencies = _context.FromFloats(_ropeFrequencies, _ropeFrequencies.Length);
        float[] identity = new float[_ropeFrequencies.Length];
        using var noRotation = _context.FromFloats(identity, identity.Length);
        try
        {
            void Project(int first, int end)
            {
                for (int start = first; start < end; start += chunk)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    int count = Math.Min(chunk, end - start);
                    using var input = count == 1 ? x[start].View(x[start].Sizes) : Concatenate(x.GetRange(start, count));
                    using var y = prenorm(input);
                    using var qRaw = weights.Q.Forward(y);
                    using var kRaw = weights.K.Forward(y);
                    var vChunk = weights.V.Forward(y);
                    Tensor qChunk = null, kChunk = null;
                    if (padW > 0)
                    {
                        var plain = vChunk;
                        try { vChunk = Ltx25DiffusionVaeKernels.Rope(_context, plain, noRotation, identity, height, width, start, heads, headDim, split, 1f, padW); }
                        finally { plain.Dispose(); }
                    }
                    try
                    {
                        long rows = qRaw.Sizes[0] * heads;
                        using (var qHeads = qRaw.View(rows, headDim))
                        using (var qNormed = RmsNorm(qHeads, weights.QNorm))
                        using (var qTokens = qNormed.View(qRaw.Sizes))
                            qChunk = Ltx25DiffusionVaeKernels.Rope(_context, qTokens, frequencies, _ropeFrequencies, height, width, start, heads, headDim, split, scale);
                        using (var kHeads = kRaw.View(rows, headDim))
                        using (var kNormed = RmsNorm(kHeads, weights.KNorm))
                        using (var kTokens = kNormed.View(kRaw.Sizes))
                            kChunk = Ltx25DiffusionVaeKernels.Rope(_context, kTokens, frequencies, _ropeFrequencies, height, width, start, heads, headDim, split, 1f, padW);
                        Evaluate(qChunk); Evaluate(kChunk); Evaluate(vChunk);
                        var qs = Split(qChunk, count); var ks = Split(kChunk, count); var vs = Split(vChunk, count);
                        for (int i = 0; i < count; ++i) { q[start + i] = qs[i]; k[start + i] = ks[i]; v[start + i] = vs[i]; }
                    }
                    finally { qChunk?.Dispose(); kChunk?.Dispose(); vChunk.Dispose(); }
                }
            }
            for (int t0 = 0; t0 < frames; t0 += chunk)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int t1 = Math.Min(frames, t0 + chunk);
                int ks = Ltx25DiffusionVaeKernels.WindowStart(t0, kt, frames);
                int ke = Ltx25DiffusionVaeKernels.WindowStart(t1 - 1, kt, frames) + kt;
                if (ke > computed) { Project(computed, ke); computed = ke; }
                using var queries = t1 - t0 == 1 ? q[t0].View(q[t0].Sizes) : Concatenate(q[t0..t1]);
                using var keys = ke - ks == 1 ? k[ks].View(k[ks].Sizes) : Concatenate(k[ks..ke]);
                using var values = ke - ks == 1 ? v[ks].View(v[ks].Sizes) : Concatenate(v[ks..ke]);
                using var attended = _context.NewF32(queries.Sizes[0], channels);
                Ltx25DiffusionVaeKernels.Attention(_context, attended, queries, keys, values, frames, height, width, t0, ks, heads, headDim, kernel, padW);
                using var projected = weights.Projection.Forward(attended);
                var updates = Split(projected, t1 - t0);
                try
                {
                    for (int i = 0; i < updates.Length; ++i) { Ops.Add(x[t0 + i], x[t0 + i], updates[i]); Evaluate(x[t0 + i]); }
                }
                finally { foreach (var update in updates) update.Dispose(); }
                // Release queries already used and keys no later window reaches.
                int nextStart = t1 < frames ? Ltx25DiffusionVaeKernels.WindowStart(t1, kt, frames) : frames;
                for (int f = t0; f < t1; ++f) { q[f]?.Dispose(); q[f] = null; }
                for (int f = 0; f < nextStart; ++f) { k[f]?.Dispose(); k[f] = null; v[f]?.Dispose(); v[f] = null; }
            }
        }
        finally
        {
            foreach (var tensor in q) tensor?.Dispose();
            foreach (var tensor in k) tensor?.Dispose();
            foreach (var tensor in v) tensor?.Dispose();
        }
    }

    /// <summary>x += w_down(silu(w_gate(y)) * w_up(y)), y = prenorm(x), in token chunks.</summary>
    private void MlpResidual(List<Tensor> x, string block, int channels, Func<Tensor, Tensor> prenorm, string gate,
        CancellationToken cancellationToken)
    {
        using var gateProjection = LoadLinear(block + ".mlp.w_gate");
        using var up = LoadLinear(block + ".mlp.w_up");
        using var down = LoadLinear(block + ".mlp.w_down", gate);
        long hidden = Hidden(channels);
        long rowsPerChunk = Math.Max(1, ChunkBudgetBytes / (hidden * sizeof(float)));
        foreach (var frame in x)
        {
            long rows = frame.Sizes[0];
            for (long start = 0; start < rows; start += rowsPerChunk)
            {
                cancellationToken.ThrowIfCancellationRequested();
                long count = Math.Min(rowsPerChunk, rows - start);
                // A whole-frame update replaces the frame's MLX array instead of copying it per slice.
                using var slice = count == rows ? frame.View(frame.Sizes) : frame.Narrow(0, start, count);
                using var y = prenorm(slice);
                using var gated = gateProjection.Forward(y);
                using var activated = DirectOps.Silu(_context, gated);
                using var lifted = up.Forward(y);
                Ops.Mul(activated, activated, lifted);
                using var output = down.Forward(activated);
                Ops.Add(slice, slice, output);
            }
            Evaluate(frame);
        }
    }

    /// <summary>Original NABlock: x += attn(norm1(x)); x += mlp(norm2(x)).</summary>
    private void DetBlock(List<Tensor> x, int height, int width, string block, int channels, int[] kernel, CancellationToken cancellationToken)
    {
        using (var weights = LoadAttention(block, channels, diffusion: false))
        using (var norm1 = _context.FromFloats(Read("decoder." + block + ".norm1.weight"), channels))
            AttentionResidual(x, height, width, channels, kernel, input => RmsNorm(input, norm1), weights, cancellationToken);
        using var norm2 = _context.FromFloats(Read("decoder." + block + ".norm2.weight"), channels);
        MlpResidual(x, block, channels, input => RmsNorm(input, norm2), null, cancellationToken);
    }

    /// <summary>LinearPixelShuffleUpsample: per frame Linear to c*st*sh*sw, then
    /// "(c p1 p2 p3) -> (t p1) (h p2) (w p3) c"; the first output frame is dropped
    /// when the stride doubles time, preserving the causal 1:2 frame mapping.</summary>
    private List<Tensor> Upsample(List<Tensor> x, int height, int width, int channels, int stage, Linear projection, int dropFirst,
        int maxFrames, CancellationToken cancellationToken, Linear after = null)
    {
        var stride = _config.Upsamples[stage].Stride;
        int st = stride[0], sh = stride[1], sw = stride[2], output = projection.Output / (st * sh * sw);
        var result = new List<Tensor>();
        try
        {
            for (int frame = 0; frame < x.Count; ++frame)
            {
                cancellationToken.ThrowIfCancellationRequested();
                using var expanded = projection.Forward(x[frame]);
                using var split = expanded.View(height, width, output, st, sh, sw);
                for (int p1 = 0; p1 < st; ++p1)
                {
                    if (frame * st + p1 < dropFirst) continue;
                    if (result.Count == maxFrames) break;
                    using var slab = split.Select(3, p1);
                    using var arranged = slab.Permute(0, 3, 1, 4, 2);
                    using var contiguous = Ops.NewContiguous(arranged);
                    var frameTensor = contiguous.View((long)height * sh * width * sw, output);
                    if (after != null)
                    {
                        var shuffled = frameTensor;
                        try { frameTensor = after.Forward(shuffled); }
                        finally { shuffled.Dispose(); }
                    }
                    Evaluate(frameTensor);
                    result.Add(frameTensor);
                }
            }
            return result;
        }
        catch { foreach (var tensor in result) tensor.Dispose(); throw; }
    }

    private void Run(float[] latents, int frames, int height, int width, Func<int, float[]> noiseFrame,
        Action<int, float[]> emitFrame, CancellationToken cancellationToken, Action<string, float[], int[]>? trace)
    {
        ArgumentNullException.ThrowIfNull(latents);
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            var geometry = Geometry(frames, height, width);
            int channels = LatentChannels;
            if (latents.Length != checked(channels * frames * height * width))
                throw new ArgumentException("Latents must be channel-major [LatentChannels,T,H,W].", nameof(latents));
            foreach (float value in latents)
                if (!float.IsFinite(value)) throw new ArgumentException("LTX VAE latents must be finite.", nameof(latents));
            var c = _config.StageChannels;
            int t = geometry.Padded[0] + geometry.Ghost, h = geometry.Padded[1], w = geometry.Padded[2];
            float[] mean = Read("per_channel_statistics.mean-of-means"), std = Read("per_channel_statistics.std-of-means");
            var x = new List<Tensor>();
            try
            {
                // Floor padding (edge replicate), ghost frames, un-normalize, conv_in.
                using (var convIn = LoadLinear("conv_in"))
                    for (int frame = 0; frame < t; ++frame)
                    {
                        int source = Math.Min(frame, frames - 1);
                        var rows = new float[h * w * channels];
                        for (int row = 0; row < h; ++row)
                        for (int column = 0; column < w; ++column)
                        {
                            int sy = Math.Clamp(row - geometry.Before[1], 0, height - 1), sx = Math.Clamp(column - geometry.Before[2], 0, width - 1);
                            for (int channel = 0; channel < channels; ++channel)
                                rows[(row * w + column) * channels + channel] =
                                    latents[((long)channel * frames + source) * height * width + sy * width + sx] * std[channel] + mean[channel];
                        }
                        using var input = _context.FromFloats(rows, h * w, channels);
                        var projected = convIn.Forward(input);
                        Evaluate(projected);
                        x.Add(projected);
                    }
                for (int stage = 0; stage < 3; ++stage)
                {
                    for (int block = 0; block < _config.StageDepths[stage]; ++block)
                        DetBlock(x, h, w, $"det_stages.{stage}.{block}", c[stage], _config.StageKernels[stage], cancellationToken);
                    var stride = _config.Upsamples[stage].Stride;
                    List<Tensor> next;
                    using (var projection = LoadLinear($"upsamples.{stage}.proj"))
                        next = Upsample(x, h, w, c[stage], stage, projection, stride[0] == 2 ? 1 : 0, int.MaxValue, cancellationToken);
                    foreach (var frame in x) frame.Dispose();
                    x = next; h *= stride[1]; w *= stride[2];
                }
                Trace(trace, "feat_stage4_input", x, h, w);
                // Stage 4 runs on the ghost-padded volume, then the ghost is cropped at
                // stage-4 resolution (the original's deferred pathway), keeping at least
                // the diffusion window's frames.
                for (int block = 0; block < _config.StageDepths[3]; ++block)
                    DetBlock(x, h, w, $"det_stages.3.{block}", c[3], _config.StageKernels[3], cancellationToken);
                var lastStride = _config.Upsamples[3].Stride;
                int timeScale = TemporalScale / lastStride[0];
                int windowFrames = Math.Max(1, (_config.Stage5Kernel[0] + lastStride[0] - 1) / lastStride[0]);
                int keep = Math.Min(x.Count, Math.Max(Math.Max(x.Count - geometry.Ghost * timeScale, 1), windowFrames));
                for (int frame = keep; frame < x.Count; ++frame) x[frame].Dispose();
                x.RemoveRange(keep, x.Count - keep);
                Trace(trace, "context", x, h, w);
                Diffusion(x, h, w, geometry, noiseFrame, emitFrame, cancellationToken);
            }
            finally { foreach (var frame in x) frame.Dispose(); }
        }
    }

    private void Trace(Action<string, float[], int[]>? trace, string name, List<Tensor> frames, int height, int width)
    {
        if (trace == null) return;
        int channels = checked((int)frames[0].Sizes[1]);
        var values = new float[checked((long)frames.Count * height * width * channels)];
        for (int frame = 0; frame < frames.Count; ++frame)
            frames[frame].GetElementsAsFloat(height * width * channels).CopyTo(values, (long)frame * height * width * channels);
        trace(name, values, [frames.Count, height, width, channels]);
    }

    /// <summary>Stage five: one x0 prediction from pixel noise at t = 1.</summary>
    private void Diffusion(List<Tensor> context, int height, int width, DecodeGeometry geometry,
        Func<int, float[]> noiseFrame, Action<int, float[]> emitFrame, CancellationToken cancellationToken)
    {
        int c5 = _config.Stage5Channels, patch = _config.PatchSize, pixelChannels = _config.PixelChannels;
        var lastStride = _config.Upsamples[3].Stride;
        int h5 = height * lastStride[1], w5 = width * lastStride[2], f5 = geometry.CanvasFrames;
        if (h5 * patch != geometry.CanvasHeight || w5 * patch != geometry.CanvasWidth)
            throw new InvalidOperationException("LTX diffusion decoder canvas geometry is inconsistent.");
        float[] modulation = Modulation();
        var x = new List<Tensor>(f5);
        try
        {
            using (var inputProjection = LoadLinear("conv_in_x_t"))
                for (int frame = 0; frame < f5; ++frame)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    // patchify "c (h q) (w r) -> (c r q) h w": channel = c*p*p + r*p + q.
                    float[] noise = noiseFrame(frame);
                    int canvasW = w5 * patch, canvasArea = h5 * patch * canvasW;
                    var rows = new float[h5 * w5 * pixelChannels];
                    for (int y = 0; y < h5; ++y)
                    for (int x0 = 0; x0 < w5; ++x0)
                    for (int channel = 0; channel < 3; ++channel)
                    for (int r = 0; r < patch; ++r)
                    for (int q = 0; q < patch; ++q)
                        rows[(y * w5 + x0) * pixelChannels + (channel * patch + r) * patch + q] =
                            noise[channel * canvasArea + (y * patch + q) * canvasW + x0 * patch + r];
                    using var input = _context.FromFloats(rows, h5 * w5, pixelChannels);
                    var projected = inputProjection.Forward(input);
                    Evaluate(projected);
                    x.Add(projected);
                }
            for (int block = 0; block < _config.DiffusionBlocks; ++block)
                DiffusionBlock(x, context, height, width, h5, w5, block, modulation, cancellationToken);
            using var normOut = _context.FromFloats(Read("decoder.norm_out.weight"), c5);
            using var output = LoadLinear("conv_out");
            int top = geometry.Before[1] * SpatialScale, left = geometry.Before[2] * SpatialScale;
            for (int frame = 0; frame < Math.Min(f5, geometry.ContentFrames); ++frame)
            {
                cancellationToken.ThrowIfCancellationRequested();
                using var normed = RmsNorm(x[frame], normOut);
                using var pixels = output.Forward(normed);
                float[] values = pixels.GetElementsAsFloat(h5 * w5 * pixelChannels);
                emitFrame(frame, Unpatchify(values, h5, w5, top, left, geometry.ContentHeight, geometry.ContentWidth));
            }
        }
        finally { foreach (var frame in x) frame.Dispose(); }
    }

    /// <summary>"(c r q) h w -> c (h q) (w r)", cropped to the content rows and columns.</summary>
    private float[] Unpatchify(float[] values, int h5, int w5, int top, int left, int height, int width)
    {
        int patch = _config.PatchSize, pixelChannels = _config.PixelChannels;
        var result = new float[3 * height * width];
        for (int channel = 0; channel < 3; ++channel)
            for (int y = 0; y < height; ++y)
            {
                int py = top + y, row = py / patch, q = py % patch;
                for (int x = 0; x < width; ++x)
                {
                    int px = left + x, column = px / patch, r = px % patch;
                    result[(channel * height + y) * width + x] = values[(row * w5 + column) * pixelChannels + (channel * patch + r) * patch + q];
                }
            }
        return result;
    }

    /// <summary>shared_adaln(t_embedder(1000 * t)) at t = 1: [7, c5], before each
    /// block's scale_shift_table. Computed on the host in Float64 accumulation.</summary>
    private float[] Modulation()
    {
        float[] features = Ltx25Denoiser.TimestepFeatures([1f], _config.TimestepScale);
        float[] Dense(float[] input, string name, bool silu)
        {
            float[] weight = Read("decoder." + name + ".weight"), bias = Read("decoder." + name + ".bias");
            int output = bias.Length, inputs = input.Length;
            var result = new float[output];
            for (int row = 0; row < output; ++row)
            {
                double sum = bias[row];
                for (int column = 0; column < inputs; ++column) sum += (double)weight[row * inputs + column] * input[column];
                result[row] = (float)sum;
            }
            if (silu) for (int i = 0; i < output; ++i) result[i] = result[i] / (1f + MathF.Exp(-result[i]));
            return result;
        }
        // TimestepEmbedding: linear_1, SiLU, linear_2; AdaLNZero: proj(silu(t_emb)).
        float[] hidden = Dense(features, "t_embedder.mlp.0", silu: true);
        float[] embedded = Dense(hidden, "t_embedder.mlp.2", silu: true);
        return Dense(embedded, "shared_adaln.proj", silu: false);
    }

    /// <summary>The original deferred diffusion block: x += context_proj(upsample(context));
    /// x += NA(modulate(norm1(x))); x += SwiGLU(modulate(norm2(x))).</summary>
    private void DiffusionBlock(List<Tensor> x, List<Tensor> context, int height, int width, int h5, int w5, int block,
        float[] modulation, CancellationToken cancellationToken)
    {
        string name = $"diff_blocks.{block}";
        int c5 = _config.Stage5Channels;
        float[] table = Read("decoder." + name + ".scale_shift_table");
        float[] Chunk(int index)
        {
            var values = new float[c5];
            for (int i = 0; i < c5; ++i) values[i] = modulation[index * c5 + i] + table[index * c5 + i];
            return values;
        }
        // Chunk order: scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp, gate_ctx.
        using var scaleMsa = _context.FromFloats(Chunk(0), c5);
        using var shiftMsa = _context.FromFloats(Chunk(1), c5);
        using var scaleMlp = _context.FromFloats(Chunk(3), c5);
        using var shiftMlp = _context.FromFloats(Chunk(4), c5);
        var lastStride = _config.Upsamples[3].Stride;
        using (var projection = LoadLinear("upsamples.3.proj"))
        using (var contextProjection = LoadLinear(name + ".context_proj", name + ".gate_ctx"))
        {
            var injected = Upsample(context, height, width, _config.ContextChannels, 3, projection, lastStride[0] == 2 ? 1 : 0,
                x.Count, cancellationToken, contextProjection);
            try
            {
                if (injected.Count != x.Count) throw new InvalidOperationException("LTX diffusion decoder context does not cover the canvas.");
                for (int frame = 0; frame < x.Count; ++frame) { Ops.Add(x[frame], x[frame], injected[frame]); Evaluate(x[frame]); }
            }
            finally { foreach (var tensor in injected) tensor.Dispose(); }
        }
        Tensor Modulated(Tensor input, Tensor gain, Tensor shift, Tensor scale)
        {
            var normed = RmsNorm(input, gain);
            try { DirectOps.ModulateRows(_context, normed, normed, shift, scale, 0, normed.Sizes[0]); return normed; }
            catch { normed.Dispose(); throw; }
        }
        using (var weights = LoadAttention(name, c5, diffusion: true))
        using (var norm1 = _context.FromFloats(Read("decoder." + name + ".norm1.weight"), c5))
            // The original's default (chunked) pathway attends in W slabs whose halos at
            // the volume edges are edge-replicated: a deliberate approximation of the
            // untiled window that this decode reproduces.
            AttentionResidual(x, h5, w5, c5, _config.Stage5Kernel, input => Modulated(input, norm1, shiftMsa, scaleMsa), weights,
                cancellationToken, _config.Stage5Kernel[2] / 2);
        using var norm2 = _context.FromFloats(Read("decoder." + name + ".norm2.weight"), c5);
        MlpResidual(x, name, c5, input => Modulated(input, norm2, shiftMlp, scaleMlp), name + ".gate_mlp", cancellationToken);
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            _disposed = true; _context.Dispose(); _file.Dispose();
        }
    }
}
