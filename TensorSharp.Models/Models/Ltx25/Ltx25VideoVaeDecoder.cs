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
using TensorSharp.Runtime;

namespace TensorSharp.Models.Ltx25;

/// <summary>Deterministic convolutional LTX 2.5 video decoder. Checkpoint BF16
/// weights are widened exactly and convolutions/activations use Float32. There
/// is no diffusion decoder, temporal tiling or noise injection here.
/// All intermediate image tensors stay on the caller's CPU/CUDA/MLX allocator.
/// A convolution's weights are released after that operation to bound residency.</summary>
public sealed class Ltx25VideoVaeDecoder : IDisposable
{
    private readonly SafetensorsFile _file;
    private readonly DirectContext _context;
    private readonly Ltx25VideoVaeConfig _config;
    private readonly string _prefix;
    private readonly object _gate = new();
    private bool _disposed;
    public int LatentChannels => _config.LatentChannels;
    public int TemporalScale => _config.TemporalScale;
    public int SpatialScale => _config.SpatialScale;
    private readonly Ltx25VideoVaeOps _ops;

    public Ltx25VideoVaeDecoder(string path, IAllocator allocator)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(allocator);
        if (allocator is not (CpuAllocator or CudaAllocator or MlxAllocator))
            throw new NotSupportedException("LTX video VAE requires a CPU, CUDA or MLX direct allocator.");
        _context = new DirectContext(allocator);
        _file = new SafetensorsFile(path);
        try
        {
            _config = Ltx25VideoVaeConfig.Parse(_file.Metadata);
            _prefix = _file.HasTensor("decoder.conv_in.conv.weight") ? "decoder." : "";
            ValidateWeights();
            _ops = new Ltx25VideoVaeOps(_context, _file, _prefix);
        }
        catch { _file.Dispose(); _context.Dispose(); throw; }
    }

    /// <summary>Decode channel-major [128,T,H,W] normalized latents to contiguous
    /// frame-major [8*(T-1)+1,32*H,32*W,3] RGB floats in [0,1]. The input remains
    /// unchanged. Optional traces contain independent channel-major [C,T,H,W]
    /// snapshots and their shapes, and introduce device readbacks.</summary>
    public float[] Decode(float[] channelMajorLatents, int latentFrames, int latentHeight, int latentWidth,
        CancellationToken cancellationToken = default, Action<string, float[], int[]>? trace = null)
    {
        lock (_gate)
        {
            using var decoded = DecodeTensor(channelMajorLatents, latentFrames, latentHeight, latentWidth, cancellationToken, trace);
            float[] result = decoded.GetElementsAsFloat(checked((int)decoded.ElementCount()));
            for (int i = 0; i < result.Length; ++i)
            {
                if (!float.IsFinite(result[i])) throw new InvalidOperationException("LTX video decoder produced a non-finite pixel.");
                result[i] = Math.Clamp((result[i] + 1f) * .5f, 0f, 1f);
            }
            return result;
        }
    }

    /// <summary>Decode to unclamped channel-major [3,T,H,W] Float32, matching the
    /// original ConvVideoDecoder.forward output layout. Useful for independent
    /// numerical validation before RGB conversion.</summary>
    public float[] DecodeRaw(float[] channelMajorLatents, int latentFrames, int latentHeight, int latentWidth,
        CancellationToken cancellationToken = default, Action<string, float[], int[]>? trace = null)
    {
        lock (_gate)
        {
            using var decoded = DecodeTensor(channelMajorLatents, latentFrames, latentHeight, latentWidth, cancellationToken, trace);
            float[] result = Planar(decoded);
            foreach (float value in result)
                if (!float.IsFinite(value)) throw new InvalidOperationException("LTX video decoder produced a non-finite value.");
            return result;
        }
    }

    private void ValidateWeights()
    {
        void Shape(string name, params long[] expected)
        {
            if (!_file.TensorShape(name).SequenceEqual(expected))
                throw new InvalidDataException($"LTX video VAE tensor '{name}' must have shape [{string.Join(',', expected)}].");
            if (_file.GetInfo(name).Dtype is not (SafetensorDtype.BF16 or SafetensorDtype.F16 or SafetensorDtype.F32))
                throw new NotSupportedException($"LTX video VAE tensor '{name}' must use BF16, F16 or Float32 storage.");
        }
        void Conv(string name, int input, int output)
        { Shape(_prefix + name + ".weight", output, input, 3, 3, 3); Shape(_prefix + name + ".bias", output); }
        Shape("per_channel_statistics.mean-of-means", LatentChannels);
        Shape("per_channel_statistics.std-of-means", LatentChannels);
        int channels = _config.BottleneckChannels;
        Conv("conv_in.conv", LatentChannels, channels);
        int index = 0;
        foreach (var block in _config.Blocks.Reverse())
        {
            string path = "up_blocks." + index++;
            if (block.Kind == "res_x")
                for (int layer = 0; layer < block.Layers; ++layer)
                {
                    Conv(path + ".res_blocks." + layer + ".conv1.conv", channels, channels);
                    Conv(path + ".res_blocks." + layer + ".conv2.conv", channels, channels);
                }
            else
            {
                int factor = block.Kind == "compress_all" ? 8 : block.Kind == "compress_space" ? 4 : 2;
                if (channels % block.Multiplier != 0)
                    throw new InvalidDataException("LTX video VAE upsample channels do not divide by the configured multiplier.");
                Conv(path + ".conv.conv", channels, checked(channels / block.Multiplier * factor));
                channels /= block.Multiplier;
            }
        }
        if (channels != _config.BaseChannels) throw new InvalidDataException("LTX VAE channel schedule is inconsistent.");
        Conv("conv_out.conv", channels, checked(3 * _config.PatchSize * _config.PatchSize));
    }

    private Tensor DecodeTensor(float[] input, int frames, int height, int width, CancellationToken cancellationToken,
        Action<string, float[], int[]>? trace)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(input);
        if (frames <= 0 || height <= 0 || width <= 0) throw new ArgumentOutOfRangeException(nameof(frames), "Latent dimensions must be positive.");
        int count = checked(LatentChannels * frames * height * width);
        _ = checked((TemporalScale * (frames - 1) + 1) * (SpatialScale * height) * (SpatialScale * width) * 3);
        if (input.Length != count) throw new ArgumentException("Latents must be channel-major [LatentChannels,T,H,W].", nameof(input));
        foreach (float value in input)
            if (!float.IsFinite(value)) throw new ArgumentException("LTX VAE latents must be finite.", nameof(input));
        cancellationToken.ThrowIfCancellationRequested();
        Tensor current;
        using (var planar = _context.FromFloats(input, LatentChannels, frames, height, width))
        using (var permuted = planar.Permute(1, 2, 3, 0)) current = Contiguous(permuted);
        try
        {
            using (var flattened = current.View((long)frames * height * width, LatentChannels))
            using (var scale = _context.FromFloats(_file.ReadFloat32("per_channel_statistics.std-of-means"), LatentChannels))
            using (var mean = _context.FromFloats(_file.ReadFloat32("per_channel_statistics.mean-of-means"), LatentChannels))
            {
                DirectOps.MulColsRows(_context, flattened, scale);
                DirectOps.AddBiasRows(_context, flattened, mean);
            }
            Trace("denormalized", current, trace);
            Replace(ref current, Convolve(current, "conv_in.conv", cancellationToken));
            Trace("conv_in", current, trace);
            int index = 0;
            foreach (var block in _config.Blocks.Reverse())
            {
                cancellationToken.ThrowIfCancellationRequested();
                string path = "up_blocks." + index++;
                if (block.Kind == "res_x")
                {
                    for (int layer = 0; layer < block.Layers; ++layer)
                    {
                        string p = path + ".res_blocks." + layer;
                        using var conv2 = ResidualBranch(current, p, cancellationToken);
                        Ops.Add(current, current, conv2);
                        // MLX's lazy sum otherwise keeps the completed branch and
                        // previous current buffer alive into the next layer.
                        if (_context.IsMlx) MlxLtx25VaeOps.Materialize(current);
                        Trace(p, current, trace);
                    }
                }
                else
                {
                    using var conv = Convolve(current, path + ".conv.conv", cancellationToken);
                    int st = block.Kind is "compress_all" or "compress_time" ? 2 : 1;
                    int sh = block.Kind is "compress_all" or "compress_space" ? 2 : 1;
                    Replace(ref current, Upsample(conv, st, sh));
                }
                Trace(path, current, trace);
            }
            using (var activated = NormSilu(current))
                Replace(ref current, Convolve(activated, "conv_out.conv", cancellationToken));
            Trace("conv_out", current, trace);
            Replace(ref current, Unpatchify(current, _config.PatchSize));
            Trace("decoded", current, trace);
            var result = current; current = null; return result;
        }
        finally { current?.Dispose(); }
    }

    private Tensor NormSilu(Tensor input) => _ops.NormSilu(input);

    private Tensor ResidualBranch(Tensor input, string name, CancellationToken cancellationToken)
    {
        // A full-resolution activation is several GiB for a six-second clip.
        // Keep only the branch's current tensor, releasing each completed
        // convolution/activation before constructing the next full-clip output.
        Tensor current = null;
        try
        {
            using (var activated = NormSilu(input))
                current = Convolve(activated, name + ".conv1.conv", cancellationToken);
            Replace(ref current, NormSilu(current));
            // Evaluating now severs the lazy activation's dependency on conv1,
            // so conv2 does not overlap both complete branch intermediates.
            if (_context.IsMlx) MlxLtx25VaeOps.Materialize(current);
            Replace(ref current, Convolve(current, name + ".conv2.conv", cancellationToken));
            var result = current; current = null; return result;
        }
        finally { current?.Dispose(); }
    }

    private Tensor Upsample(Tensor input, int temporalStride, int spatialStride)
    {
        long t = input.Sizes[0], h = input.Sizes[1], w = input.Sizes[2];
        long channels = input.Sizes[3] / (temporalStride * spatialStride * spatialStride);
        if (_context.IsCuda)
        {
            // The existing native strided-copy kernel supports at most four
            // axes. Factor the seven-axis shuffle into supported permutations;
            // a generic seven-axis Copy would silently read back to the CPU.
            using var grouped = input.View(t * h * w, channels, temporalStride * spatialStride * spatialStride);
            using var channelsLast = grouped.Permute(0, 2, 1);
            using var reordered = Contiguous(channelsLast);
            using var timeGroups = reordered.View(t, h * w, temporalStride, spatialStride * spatialStride * channels);
            using var timeFirst = timeGroups.Permute(0, 2, 1, 3);
            using var timeContiguous = Contiguous(timeFirst);
            using var heightGroups = timeContiguous.View(t * temporalStride * h, w, spatialStride, spatialStride * channels);
            using var heightFirst = heightGroups.Permute(0, 2, 1, 3);
            using var spatialContiguous = Contiguous(heightFirst);
            using var result = spatialContiguous.View(t * temporalStride, h * spatialStride, w * spatialStride, channels);
            if (temporalStride == 1) return result.View(result.Sizes);
            using var cropped = result.Narrow(0, 1, t * temporalStride - 1);
            return Contiguous(cropped);
        }
        using var split = input.View(t, h, w, channels, temporalStride, spatialStride, spatialStride);
        using var shuffled = split.Permute(0, 4, 1, 5, 2, 6, 3);
        using var contiguous = Contiguous(shuffled);
        using var expanded = contiguous.View(t * temporalStride, h * spatialStride, w * spatialStride, channels);
        if (temporalStride == 1) return expanded.View(expanded.Sizes);
        // Original DepthToSpaceUpsample always discards the first expanded frame.
        using var trimmed = expanded.Narrow(0, 1, t * temporalStride - 1);
        return Contiguous(trimmed);
    }

    private Tensor Unpatchify(Tensor input, int patch)
    {
        long t = input.Sizes[0], h = input.Sizes[1], w = input.Sizes[2];
        if (_context.IsCuda)
        {
            using var groups = input.View(t * h * w, 3, patch, patch);
            using var pixelChannelsLast = groups.Permute(0, 3, 2, 1);
            using var pixels = Contiguous(pixelChannelsLast);
            using var rows = pixels.View(t * h, w, patch, patch * 3);
            using var heightFirst = rows.Permute(0, 2, 1, 3);
            using var contiguousPixels = Contiguous(heightFirst);
            return contiguousPixels.View(t, h * patch, w * patch, 3);
        }
        // Unlike upsampling's (c,pt,ph,pw), patchify packs (c,pw,ph).
        using var split = input.View(t, h, w, 3, patch, patch);
        using var permuted = split.Permute(0, 1, 5, 2, 4, 3);
        using var contiguous = Contiguous(permuted);
        return contiguous.View(t, h * patch, w * patch, 3);
    }

    private Tensor Convolve(Tensor input, string name, CancellationToken cancellationToken)
        => _ops.Convolve(input, name, cancellationToken);
    private Tensor Contiguous(Tensor source) => _ops.Contiguous(source);
    private float[] Planar(Tensor tensor) => _ops.Planar(tensor);
    private void Trace(string name, Tensor tensor, Action<string, float[], int[]>? trace)
    {
        if (trace == null) return;
        trace(name, Planar(tensor), [checked((int)tensor.Sizes[3]), checked((int)tensor.Sizes[0]),
            checked((int)tensor.Sizes[1]), checked((int)tensor.Sizes[2])]);
    }
    private static void Replace(ref Tensor current, Tensor replacement)
    { var previous = current; current = replacement; previous.Dispose(); }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            _disposed = true; _context.Dispose(); _file.Dispose();
        }
    }
}
