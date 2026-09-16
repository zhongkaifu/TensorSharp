// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using TensorSharp.GGML;

namespace TensorSharp.Models;

/// <summary>
/// NVIDIA Nemotron Omni's Parakeet FastConformer and sound projector. The audio
/// companion retains the official sound_encoder/sound_projection tensor names.
/// Each clip is encoded separately, so padding from a neighbouring clip cannot
/// change its length or leak into its bidirectional attention.
/// </summary>
public sealed class NemotronAudioEncoder : IDisposable
{
    private const string Encoder = "sound_encoder.encoder.";
    private readonly IAllocator _allocator;
    private readonly Dictionary<string, float[]> _vectors = new(StringComparer.Ordinal);
    private readonly Dictionary<string, Tensor> _matrices = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int[]> _shapes = new(StringComparer.Ordinal);
    private readonly int _hidden, _heads, _layers, _channels, _kernel, _ffn;
    private readonly bool _bf16;
    private readonly Action<string, float[]> _trace;
    private ModelBase _host;
    public int MelBins { get; }
    public int ProjectionDim { get; }
    public void SetHostModel(ModelBase model) => _host = model;

    public NemotronAudioEncoder(string path, IAllocator allocator)
        : this(path, allocator, null) { }

    internal NemotronAudioEncoder(string path, IAllocator allocator, Action<string, float[]> trace)
    {
        _trace = trace;
        _allocator = allocator ?? throw new ArgumentNullException(nameof(allocator));
        using var gguf = new GgufFile(path);
        int Value(string key, uint fallback) => checked((int)gguf.GetUint32("nemotron.audio." + key, fallback));
        _hidden = Value("hidden_size", 1024); _heads = Value("num_attention_heads", 8);
        _layers = Value("num_hidden_layers", 24); _channels = Value("subsampling_conv_channels", 256);
        _kernel = Value("conv_kernel_size", 9); _ffn = Value("intermediate_size", 4096);
        MelBins = Value("num_mel_bins", 128); ProjectionDim = Value("projection_dim", 2688);
        _bf16 = gguf.GetBool("nemotron.audio.compute_bf16", false);
        if (_hidden <= 0 || _heads <= 0 || _hidden % _heads != 0 || _hidden % 2 != 0 ||
            _layers <= 0 || _channels <= 0 || _kernel <= 0 || _kernel % 2 == 0 ||
            MelBins <= 0 || MelBins % 8 != 0 || ProjectionDim <= 0 || _ffn <= 0 ||
            Value("subsampling_factor", 8) != 8 || Value("subsampling_conv_stride", 2) != 2 ||
            Value("subsampling_conv_kernel_size", 3) != 3 || Value("sampling_rate", 16000) != 16000)
            throw new NotSupportedException("Unsupported Nemotron Parakeet audio encoder dimensions or sampling configuration.");
        try
        {
            foreach (var info in gguf.Tensors.Values)
            {
                if (!info.Name.StartsWith(Encoder, StringComparison.Ordinal) &&
                    !info.Name.StartsWith("sound_projection.", StringComparison.Ordinal)) continue;
                if (info.Name.Contains(".feature_extractor.", StringComparison.Ordinal)) continue;
                int[] shape = info.Shape.Reverse().Select(v => checked((int)v)).ToArray();
                _shapes.Add(info.Name, shape);
                float[] data = new float[checked((int)info.NumElements)];
                byte[] raw = gguf.ReadTensorData(info);
                if (info.Type == GgmlTensorType.F32) Buffer.BlockCopy(raw, 0, data, 0, raw.Length);
                else NativeDequant.DequantizeToFloat32((int)info.Type, raw, 0, data, 0, data.Length);
                bool matrix = info.Name.EndsWith(".weight", StringComparison.Ordinal) &&
                    (shape.Length == 2 || (shape.Length > 2 && shape.Skip(2).All(n => n == 1)));
                if (!matrix) { _vectors.Add(info.Name, data); continue; }
                int rows = shape[0], cols = shape[1];
                // Store only the transposed matrix consumed by Addmm, not a second
                // packed copy of every 600M-parameter encoder weight.
                float[] transposed = new float[data.Length];
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < cols; c++) transposed[c * rows + r] = data[r * cols + c];
                var tensor = new Tensor(_allocator, DType.Float32, cols, rows);
                try { tensor.SetElementsAsFloat(transposed); _matrices.Add(info.Name, tensor); }
                catch { tensor.Dispose(); throw; }
            }
            ValidateWeights(Value("projection_hidden_size", 4096));
        }
        catch { Dispose(); throw; }
    }

    private void Shape(string name, params int[] expected)
    {
        if (!_shapes.TryGetValue(name, out var actual) || !actual.SequenceEqual(expected))
            throw new InvalidDataException($"Missing or incompatible Nemotron audio tensor '{name}'.");
    }

    private void ValidateWeights(int projectionHidden)
    {
        Shape(Encoder + "subsampling.layers.0.weight", _channels, 1, 3, 3);
        foreach (int i in new[] { 0, 2, 3, 5, 6 }) Shape(Encoder + $"subsampling.layers.{i}.bias", _channels);
        foreach (int i in new[] { 2, 5 }) Shape(Encoder + $"subsampling.layers.{i}.weight", _channels, 1, 3, 3);
        foreach (int i in new[] { 3, 6 }) Shape(Encoder + $"subsampling.layers.{i}.weight", _channels, _channels, 1, 1);
        Shape(Encoder + "subsampling.linear.weight", _hidden, _channels * (MelBins / 8));
        Shape(Encoder + "subsampling.linear.bias", _hidden);
        for (int i = 0; i < _layers; i++)
        {
            string p = Encoder + $"layers.{i}.";
            foreach (string n in new[] { "norm_feed_forward1", "norm_feed_forward2", "norm_self_att", "norm_conv", "norm_out" })
            { Shape(p + n + ".weight", _hidden); Shape(p + n + ".bias", _hidden); }
            foreach (string n in new[] { "feed_forward1", "feed_forward2" })
            { Shape(p + n + ".linear1.weight", _ffn, _hidden); Shape(p + n + ".linear2.weight", _hidden, _ffn); }
            foreach (string n in new[] { "q_proj", "k_proj", "v_proj", "o_proj", "relative_k_proj" })
                Shape(p + "self_attn." + n + ".weight", _hidden, _hidden);
            Shape(p + "self_attn.bias_u", _heads, _hidden / _heads);
            Shape(p + "self_attn.bias_v", _heads, _hidden / _heads);
            Shape(p + "conv.pointwise_conv1.weight", 2 * _hidden, _hidden, 1);
            Shape(p + "conv.pointwise_conv2.weight", _hidden, _hidden, 1);
            Shape(p + "conv.depthwise_conv.weight", _hidden, 1, _kernel);
            foreach (string n in new[] { "weight", "bias", "running_mean", "running_var" }) Shape(p + "conv.norm." + n, _hidden);
        }
        Shape("sound_projection.norm.weight", _hidden);
        Shape("sound_projection.linear1.weight", projectionHidden, _hidden);
        Shape("sound_projection.linear2.weight", ProjectionDim, projectionHidden);
    }

    public static int OutputLength(int melFrames)
    {
        if (melFrames <= 0) throw new ArgumentOutOfRangeException(nameof(melFrames));
        return checked((melFrames - 1) / 8 + 1);
    }

    public Tensor Encode(float[] mel, int frames, int validFrames)
    {
        ArgumentNullException.ThrowIfNull(mel);
        if (frames <= 0 || validFrames <= 0 || validFrames > frames || mel.Length != checked(frames * MelBins))
            throw new ArgumentException("Audio mel dimensions and valid frame count are inconsistent.");
        if (mel.Any(v => !float.IsFinite(v))) throw new ArgumentException("Audio features must be finite.", nameof(mel));
        float[] x = (float[])mel.Clone(); Round(x);
        int time = frames, frequency = MelBins, valid = validFrames;
        // Intermediate layout is [time, frequency, channel]. Mask after every
        // convolution, including the pointwise ones with a learned bias.
        x = Subsample(x, ref time, ref frequency, ref valid, 1, 0);
        Relu(x);
        x = Subsample(x, ref time, ref frequency, ref valid, _channels, 2);
        x = Linear(x, time * frequency, Encoder + "subsampling.layers.3"); MaskSpatial(x, time, frequency, valid); Relu(x);
        x = Subsample(x, ref time, ref frequency, ref valid, _channels, 5);
        x = Linear(x, time * frequency, Encoder + "subsampling.layers.6"); MaskSpatial(x, time, frequency, valid); Relu(x);
        // HF flattens [channels, frequency] for each time, not the interleaved
        // [frequency, channels] representation used by the convolution kernels.
        float[] flattened = new float[x.Length];
        for (int t = 0; t < time; t++) for (int c = 0; c < _channels; c++) for (int f = 0; f < frequency; f++)
            flattened[t * _channels * frequency + c * frequency + f] = x[(t * frequency + f) * _channels + c];
        // A one-bin frequency axis leaves HF's [time, channel] transpose
        // noncontiguous, so its BF16 linear rounds matmul before adding bias.
        x = Linear(flattened, time, Encoder + "subsampling.linear", splitBias: frequency == 1 && time > 1);
        float[] positions = new float[checked((2 * time - 1) * _hidden)];
        for (int t = 0; t < 2 * time - 1; t++) for (int h = 0; h < _hidden; h += 2)
        {
            // inv_freq is a floating module buffer and follows the official
            // encoder's BF16 conversion before forward casts it back to F32.
            float inverseFrequency = RoundScalar(1 / MathF.Pow(10000, (float)h / _hidden));
            float angle = (time - 1 - t) * inverseFrequency;
            positions[t * _hidden + h] = MathF.Sin(angle); positions[t * _hidden + h + 1] = MathF.Cos(angle);
        }
        Round(positions);
        for (int layer = 0; layer < _layers; layer++)
        {
            string p = Encoder + $"layers.{layer}.";
            Add(x, FeedForward(Norm(x, p + "norm_feed_forward1"), time, p + "feed_forward1"), .5f);
            Add(x, Attention(Norm(x, p + "norm_self_att"), positions, time, valid, p + "self_attn"));
            Add(x, Convolution(Norm(x, p + "norm_conv"), time, valid, p + "conv"));
            Add(x, FeedForward(Norm(x, p + "norm_feed_forward2"), time, p + "feed_forward2"), .5f);
            x = Norm(x, p + "norm_out");
            _host?.YieldGpuComputeLock();
        }
        x = Norm(x, "sound_projection.norm", rms: true);
        x = Linear(x, time, "sound_projection.linear1");
        for (int i = 0; i < x.Length; i++) { float r = Math.Max(0, x[i]); x[i] = r * r; }
        Round(x);
        x = Linear(x, time, "sound_projection.linear2");
        if (x.Any(v => !float.IsFinite(v))) throw new InvalidOperationException("Nemotron audio encoder produced non-finite embeddings.");
        var result = new Tensor(_allocator, DType.Float32, time, ProjectionDim);
        try { result.SetElementsAsFloat(x); return result; }
        catch { result.Dispose(); throw; }
    }

    private float[] Linear(float[] x, int rows, string prefix, bool splitBias = false)
    {
        Tensor w = _matrices[prefix + ".weight"];
        int input = checked((int)w.Sizes[0]), output = checked((int)w.Sizes[1]);
        if (x.Length != checked(rows * input)) throw new InvalidDataException("Audio linear input dimensions do not match its weights.");
        using var a = new Tensor(_allocator, DType.Float32, rows, input);
        using var b = new Tensor(_allocator, DType.Float32, rows, output);
        a.SetElementsAsFloat(x); Ops.Addmm(b, 0, b, 1, a, w);
        float[] result = b.GetElementsAsFloat(checked(rows * output));
        if (splitBias) Round(result);
        if (_vectors.TryGetValue(prefix + ".bias", out var bias))
            for (int t = 0; t < rows; t++) for (int h = 0; h < output; h++) result[t * output + h] += bias[h];
        Round(result); Trace(prefix, result); return result;
    }

    private float[] Norm(float[] x, string prefix, bool rms = false)
    {
        float[] weight = _vectors[prefix + ".weight"];
        _vectors.TryGetValue(prefix + ".bias", out var bias);
        float[] y = new float[x.Length];
        for (int t = 0; t < x.Length / _hidden; t++)
        {
            double mean = 0, variance = 0;
            if (!rms) { for (int h = 0; h < _hidden; h++) mean += x[t * _hidden + h]; mean /= _hidden; }
            for (int h = 0; h < _hidden; h++) { double d = x[t * _hidden + h] - mean; variance += d * d; }
            float scale = 1 / MathF.Sqrt((float)(variance / _hidden) + 1e-5f);
            for (int h = 0; h < _hidden; h++) y[t * _hidden + h] = (x[t * _hidden + h] - (float)mean) * scale * weight[h] + (bias?[h] ?? 0);
        }
        Round(y); Trace(prefix, y); return y;
    }

    private float[] FeedForward(float[] x, int time, string prefix)
    {
        x = Linear(x, time, prefix + ".linear1"); Silu(x);
        return Linear(x, time, prefix + ".linear2");
    }

    private float[] Attention(float[] x, float[] positions, int time, int valid, string prefix)
    {
        float[] q = Linear(x, time, prefix + ".q_proj"), k = Linear(x, time, prefix + ".k_proj"), v = Linear(x, time, prefix + ".v_proj");
        float[] r = Linear(positions, 2 * time - 1, prefix + ".relative_k_proj");
        float[] u = _vectors[prefix + ".bias_u"], b = _vectors[prefix + ".bias_v"];
        float[] result = new float[x.Length]; int width = _hidden / _heads;
        // One score row at a time bounds scratch memory for long clips.
        Parallel.For(0, valid * _heads, task =>
        {
            int t = task / _heads, head = task % _heads, begin = head * width;
            float[] scores = new float[valid]; float max = float.NegativeInfinity;
            for (int s = 0; s < valid; s++)
            {
                float content = 0, relative = 0;
                int relativeRow = time - 1 - t + s;
                for (int h = begin; h < begin + width; h++)
                {
                    float qu = RoundScalar(q[t * _hidden + h] + u[h]);
                    float qb = RoundScalar(q[t * _hidden + h] + b[h]);
                    content += qu * k[s * _hidden + h]; relative += qb * r[relativeRow * _hidden + h];
                }
                float scale = 1 / MathF.Sqrt(width);
                scores[s] = content * scale + RoundScalar(RoundScalar(relative) * scale); max = Math.Max(max, scores[s]);
            }
            float sum = 0;
            for (int s = 0; s < valid; s++)
            {
                float numerator = MathF.Exp(scores[s] - max);
                sum += numerator;
                // The official BF16 SDPA path stores the unnormalized
                // exponentials in BF16 for its value product, retaining the
                // F32 denominator until the final context normalization.
                scores[s] = RoundScalar(numerator);
            }
            for (int h = begin; h < begin + width; h++)
            {
                float value = 0;
                for (int s = 0; s < valid; s++) value += scores[s] * v[s * _hidden + h];
                result[t * _hidden + h] = value / sum;
            }
        });
        Round(result); Trace(prefix + ".context", result); return Linear(result, time, prefix + ".o_proj");
    }

    private float[] Convolution(float[] x, int time, int valid, string prefix)
    {
        float[] up = Linear(x, time, prefix + ".pointwise_conv1"), gated = new float[time * _hidden];
        for (int t = 0; t < valid; t++) for (int h = 0; h < _hidden; h++)
            gated[t * _hidden + h] = up[t * 2 * _hidden + h] / (1 + MathF.Exp(-up[t * 2 * _hidden + _hidden + h]));
        Round(gated);
        float[] w = _vectors[prefix + ".depthwise_conv.weight"];
        _vectors.TryGetValue(prefix + ".depthwise_conv.bias", out var bias);
        float[] normW = _vectors[prefix + ".norm.weight"], normB = _vectors[prefix + ".norm.bias"];
        float[] mean = _vectors[prefix + ".norm.running_mean"], variance = _vectors[prefix + ".norm.running_var"];
        float[] y = new float[gated.Length];
        for (int t = 0; t < time; t++) for (int h = 0; h < _hidden; h++)
        {
            float value = bias?[h] ?? 0;
            for (int j = 0; j < _kernel; j++)
            { int source = t + j - _kernel / 2; if (source >= 0 && source < time) value += gated[source * _hidden + h] * w[h * _kernel + j]; }
            value = RoundScalar(value);
            y[t * _hidden + h] = (value - mean[h]) / MathF.Sqrt(variance[h] + 1e-5f) * normW[h] + normB[h];
        }
        Round(y); Trace(prefix + ".norm", y); Silu(y); return Linear(y, time, prefix + ".pointwise_conv2");
    }

    private float[] Subsample(float[] x, ref int time, ref int frequency, ref int valid, int inputChannels, int layer)
    {
        string p = Encoder + $"subsampling.layers.{layer}";
        float[] w = _vectors[p + ".weight"], bias = _vectors[p + ".bias"];
        int oldTime = time, oldFrequency = frequency, newTime = (time + 1) / 2, newFrequency = (frequency + 1) / 2;
        int newValid = (valid + 1) / 2;
        float[] y = new float[checked(newTime * newFrequency * _channels)];
        Parallel.For(0, newValid, t =>
        {
            for (int f = 0; f < newFrequency; f++) for (int c = 0; c < _channels; c++)
            {
                float value = bias[c];
                for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++)
                {
                    int st = 2 * t + ky - 1, sf = 2 * f + kx - 1;
                    if (st >= 0 && st < oldTime && sf >= 0 && sf < oldFrequency)
                        value += x[(st * oldFrequency + sf) * inputChannels + (inputChannels == 1 ? 0 : c)] * w[c * 9 + ky * 3 + kx];
                }
                y[(t * newFrequency + f) * _channels + c] = value;
            }
        });
        time = newTime; frequency = newFrequency; valid = newValid; Round(y); return y;
    }

    private void MaskSpatial(float[] x, int time, int frequency, int valid)
        => Array.Clear(x, checked(valid * frequency * _channels), checked((time - valid) * frequency * _channels));
    private void Add(float[] x, float[] other, float scale = 1)
    { for (int i = 0; i < x.Length; i++) x[i] += scale * other[i]; Round(x); }
    private void Silu(float[] x)
    { for (int i = 0; i < x.Length; i++) x[i] /= 1 + MathF.Exp(-x[i]); Round(x); }
    private static void Relu(float[] x)
    { for (int i = 0; i < x.Length; i++) x[i] = Math.Max(0, x[i]); }
    private void Round(float[] x)
    { if (_bf16) for (int i = 0; i < x.Length; i++) x[i] = RoundScalar(x[i]); }
    private void Trace(string name, float[] values) => _trace?.Invoke(name, (float[])values.Clone());
    private float RoundScalar(float x)
    {
        if (!_bf16) return x;
        uint bits = BitConverter.SingleToUInt32Bits(x);
        return BitConverter.UInt32BitsToSingle((bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u);
    }
    public void Dispose()
    { foreach (var value in _matrices.Values) value.Dispose(); _matrices.Clear(); _vectors.Clear(); _shapes.Clear(); }
}
