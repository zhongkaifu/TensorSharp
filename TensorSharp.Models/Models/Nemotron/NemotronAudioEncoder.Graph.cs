// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Linq;

using TensorSharp.GGML;

using TensorSharp.Graph;
using TensorSharp.Models.Graph;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

/// <summary>
/// The Parakeet FastConformer and sound projector as one graph program per clip, on the executor of the model's allocator
/// (GGML cpu / cuda / metal / vulkan, the managed CPU, MLX, direct CUDA), for a companion that computes in F32
/// (<c>nemotron.audio.compute_bf16</c> false; a BF16-emulating companion keeps the managed path, which rounds every
/// intermediate as the official BF16 encoder does). The program is the managed path's arithmetic, in the layout ggml's
/// feature maps take:
/// <list type="bullet">
/// <item>Subsampling over [channels, mel, time]: Conv2d(1 -> C, 3x3, stride 2, pad 1), ReLU, then twice a depthwise 3x3 /
/// stride 2 conv and a pointwise one with ReLU; after each conv the time rows past the clip's valid frames are zeros (the
/// managed path computes only the valid ones). The map is flattened channel-major (c * mel + f) and projected.</item>
/// <item>Each conformer layer: half a feed-forward (LayerNorm, linear, SiLU, linear), relative-position self-attention
/// (Transformer-XL: content scores of q + bias_u against the keys, position scores of q + bias_v against the projected
/// sinusoid table, shifted so query t reads relative position t - s; keys past the valid frames masked, query rows past
/// them zero), the conv module (pointwise to 2H, GLU, a depthwise conv over time with 'same' zero padding, inference
/// BatchNorm folded into a scale and shift, SiLU, pointwise), the other half feed-forward, and the layer's LayerNorm.</item>
/// <item>Projection: RMSNorm, linear, ReLU², linear.</item>
/// </list>
/// </summary>
public sealed partial class NemotronAudioEncoder
{
    private NetworkWeights _graphWeights;
    private NetworkGraph _graph;

    /// <summary>Where the graph path's embeddings are handed back (the model's tower allocator: host memory on direct CUDA,
    /// as the managed path's were); null: the weights' allocator.</summary>
    private IAllocator _outputAllocator;

    /// <summary>True when clips encode as graph programs (<see cref="EncodeGraph"/>).</summary>
    public bool RunsAsGraphs => _graph != null;

    /// <summary>The graph programs' plan cache and executor (tests).</summary>
    internal NetworkGraph Graph => _graph;

    /// <summary>
    /// The companion at <paramref name="path"/>: an F32-compute one as graph programs on <paramref name="allocator"/>'s executor
    /// (the model's), a BF16-emulating one on the managed path over <paramref name="hostAllocator"/>.
    /// </summary>
    public static NemotronAudioEncoder Load(string path, IAllocator allocator, BackendType backend, IAllocator hostAllocator)
    {
        bool bf16;
        using (var gguf = new GgufFile(path))
            bf16 = gguf.GetBool("nemotron.audio.compute_bf16", false);
        if (bf16 || GraphExecutors.For(allocator) == null)
            return new NemotronAudioEncoder(path, hostAllocator ?? allocator, null);
        return new NemotronAudioEncoder(path, allocator, null, backend) { _outputAllocator = hostAllocator };
    }

    private void LoadGraphWeights(GgufFile gguf, IAllocator allocator, BackendType backend, IGraphExecutor executor)
    {
        _graphWeights = new NetworkWeights(allocator, backend);
        foreach (GgufTensorInfo info in gguf.Tensors.Values)
        {
            if (!_shapes.TryGetValue(info.Name, out int[] shape)) continue;
            // The products' matrices (the managed path's rule: a .weight of [out, in] and trailing ones) keep their file
            // encoding where the backend has a kernel for it; the vectors, biases and conv kernels are F32 values.
            bool matrix = info.Name.EndsWith(".weight", StringComparison.Ordinal) &&
                (shape.Length == 2 || (shape.Length > 2 && shape.Skip(2).All(n => n == 1)));
            if (matrix)
            {
                _graphWeights.LoadTensor(gguf, info.Name);
                continue;
            }
            var values = new float[checked((int)info.NumElements)];
            byte[] raw = gguf.ReadTensorData(info);
            if (info.Type == GgmlTensorType.F32) Buffer.BlockCopy(raw, 0, values, 0, raw.Length);
            else NativeDequant.DequantizeToFloat32((int)info.Type, raw, 0, values, 0, values.Length);
            _graphWeights.AddF32(info.Name, values, Array.ConvertAll(info.Shape, d => (long)d));
        }
        // Inference BatchNorm as a scale and shift: (v - mean) / sqrt(var + eps) * w + b = v * s + (b - mean * s).
        for (int i = 0; i < _layers; i++)
        {
            string p = Encoder + $"layers.{i}.conv.norm.";
            float[] w = _graphWeights.ReadF32(p + "weight"), b = _graphWeights.ReadF32(p + "bias");
            float[] mean = _graphWeights.ReadF32(p + "running_mean"), variance = _graphWeights.ReadF32(p + "running_var");
            var scale = new float[_hidden];
            var shift = new float[_hidden];
            for (int h = 0; h < _hidden; h++)
            {
                scale[h] = w[h] / MathF.Sqrt(variance[h] + 1e-5f);
                shift[h] = b[h] - mean[h] * scale[h];
            }
            _graphWeights.AddF32(p + "#scale", scale, _hidden);
            _graphWeights.AddF32(p + "#shift", shift, _hidden);
        }
        _graph = new NetworkGraph(_graphWeights, executor, "nemotron audio encoder");
    }

    /// <summary>The clip's rows [time / 8, ProjectionDim] on the encoder's allocator, as one program.</summary>
    private Tensor EncodeGraph(float[] mel, int frames, int validFrames)
    {
        int output = 0;
        float[][] outputs = _graph.Run(b => output = b.Output(Emit(b, mel, frames, validFrames)));
        float[] rows = outputs[output];
        foreach (float v in rows)
            if (!float.IsFinite(v)) throw new InvalidOperationException("Nemotron audio encoder produced non-finite embeddings.");
        var result = new Tensor(_outputAllocator ?? _graphWeights.Allocator, DType.Float32, OutputLength(frames), ProjectionDim);
        try { result.SetElementsAsFloat(rows); return result; }
        catch { result.Dispose(); throw; }
    }

    private NetValue Emit(NetworkBuilder b, float[] mel, int frames, int validFrames)
    {
        int c = _channels, time = frames, frequency = MelBins, valid = validFrames;
        NetValue x = b.Input(mel, 1, MelBins, frames);                                          // [1, mel, time]

        // Subsampling: each conv stage halves mel and time; the rows past the valid frames are zeros after every conv.
        x = b.Conv2d(x, Encoder + "subsampling.layers.0.weight", Encoder + "subsampling.layers.0.bias", 2, 2, 1, 1);
        Halve(ref time, ref frequency, ref valid);
        x = b.Relu(MaskTime(b, x, time, valid));
        x = DepthwiseStage(b, x, 2, ref time, ref frequency, ref valid);
        x = Pointwise(b, x, 3, time, valid);
        x = DepthwiseStage(b, x, 5, ref time, ref frequency, ref valid);
        x = Pointwise(b, x, 6, time, valid);
        // Channel-major features per time row (c * mel + f), as HF flattens [channels, frequency].
        x = b.Linear(b.Reshape(b.Cont(b.Permute(x, 1, 0, 2, 3)), (long)c * frequency, time),
            Encoder + "subsampling.linear.weight", Bias(Encoder + "subsampling.linear"));

        NetValue positions = b.Input(RelativePositions(time), _hidden, 2 * time - 1);
        NetValue rowMask = b.Input(TimeMask(time, valid), 1, time);
        NetValue keyMask = b.Mask(KeyMask(time, valid), time, time);
        for (int layer = 0; layer < _layers; layer++)
        {
            string p = Encoder + $"layers.{layer}.";
            x = b.Add(x, b.Scale(FeedForward(b, Norm(b, x, p + "norm_feed_forward1"), p + "feed_forward1"), 0.5f));
            x = b.Add(x, Attention(b, Norm(b, x, p + "norm_self_att"), positions, rowMask, keyMask, time, p + "self_attn"));
            x = b.Add(x, ConvModule(b, Norm(b, x, p + "norm_conv"), rowMask, time, p + "conv"));
            x = b.Add(x, b.Scale(FeedForward(b, Norm(b, x, p + "norm_feed_forward2"), p + "feed_forward2"), 0.5f));
            x = Norm(b, x, p + "norm_out");
        }
        x = b.RmsNorm(x, 1e-5f, b.Vector("sound_projection.norm.weight", _hidden));
        if (Bias("sound_projection.norm") is { } normBias) x = b.Add(x, b.Vector(normBias, _hidden));
        x = b.Linear(x, "sound_projection.linear1.weight", Bias("sound_projection.linear1"));
        x = b.Relu(x);
        x = b.Mul(x, x);
        return b.Linear(x, "sound_projection.linear2.weight", Bias("sound_projection.linear2"));
    }

    private static void Halve(ref int time, ref int frequency, ref int valid)
    {
        time = (time + 1) / 2;
        frequency = (frequency + 1) / 2;
        valid = (valid + 1) / 2;
    }

    /// <summary>A depthwise 3x3 / stride 2 conv stage over [C, mel, time] and its bias, valid rows kept.</summary>
    private NetValue DepthwiseStage(NetworkBuilder b, NetValue x, int layer, ref int time, ref int frequency, ref int valid)
    {
        string p = Encoder + $"subsampling.layers.{layer}";
        NetValue y = b.Add(b.DepthwiseConv2d(x, b.Weight(p + ".weight"), 2, 2, 1, 1), b.Vector(p + ".bias", _channels));
        Halve(ref time, ref frequency, ref valid);
        return MaskTime(b, y, time, valid);
    }

    /// <summary>A pointwise conv (a [1, 1, C, C] kernel and its bias) over [C, mel, time], valid rows kept, then ReLU.</summary>
    private NetValue Pointwise(NetworkBuilder b, NetValue x, int layer, int time, int valid)
    {
        string p = Encoder + $"subsampling.layers.{layer}";
        long c = x.Ne0, mels = x.Ne1, n = x.Ne2;
        NetValue y = b.Linear(b.Reshape(x, c, mels * n), b.Weight(p + ".weight", c, _channels), b.Vector(p + ".bias", _channels));
        return b.Relu(MaskTime(b, b.Reshape(y, _channels, mels, n), time, valid));
    }

    /// <summary>[C, mel, time] with the rows from <paramref name="valid"/> on set to zero.</summary>
    private static NetValue MaskTime(NetworkBuilder b, NetValue x, int time, int valid)
        => valid >= time ? x : b.Mul(x, b.Input(TimeMask(time, valid), 1, 1, time));

    private NetValue Norm(NetworkBuilder b, NetValue x, string prefix)
        => b.LayerNorm(x, 1e-5f, b.Vector(prefix + ".weight", _hidden), Bias(prefix) is { } bias ? b.Vector(bias, _hidden) : default);

    private NetValue FeedForward(NetworkBuilder b, NetValue x, string prefix)
        => b.Linear(b.Silu(b.Linear(x, prefix + ".linear1.weight", Bias(prefix + ".linear1"))), prefix + ".linear2.weight", Bias(prefix + ".linear2"));

    private NetValue Attention(NetworkBuilder b, NetValue x, NetValue positions, NetValue rowMask, NetValue keyMask, int time, string prefix)
    {
        int width = _hidden / _heads;
        NetValue Heads(NetValue v, long rows) => b.Reshape(v, width, _heads, rows);
        NetValue q = Heads(b.Linear(x, prefix + ".q_proj.weight", Bias(prefix + ".q_proj")), time);
        NetValue k = Heads(b.Linear(x, prefix + ".k_proj.weight", Bias(prefix + ".k_proj")), time);
        NetValue v = Heads(b.Linear(x, prefix + ".v_proj.weight", Bias(prefix + ".v_proj")), time);
        NetValue r = Heads(b.Linear(positions, prefix + ".relative_k_proj.weight", Bias(prefix + ".relative_k_proj")), 2L * time - 1);
        NetValue qu = b.Add(q, b.Weight(prefix + ".bias_u", width, _heads, 1));
        NetValue qv = b.Add(q, b.Weight(prefix + ".bias_v", width, _heads, 1));
        NetValue ByHead(NetValue t) => b.Cont(b.Permute(t, 0, 2, 1, 3));                          // [width, rows, heads]
        NetValue content = b.MatMul(ByHead(k), ByHead(qu));                                      // [keys, queries, heads]
        NetValue relative = b.MatMul(ByHead(r), ByHead(qv));                                     // [2T - 1, queries, heads]
        // Query t, key s reads relative row T - 1 - t + s: element (s, t) of a view whose query rows step 2T - 2.
        long f = sizeof(float);
        NetValue shifted = b.Cont(b.View(relative, new long[] { time, time, _heads },
            new[] { (2L * time - 2) * f, (2L * time - 1) * time * f }, (time - 1) * f));
        NetValue scores = b.Add(content, shifted);
        NetValue probabilities = b.Op(GraphOp.SoftMaxExt, stackalloc[] { scores, keyMask }, stackalloc long[] { time, time, _heads },
            default, stackalloc[] { 1f / MathF.Sqrt(width), 0f });
        NetValue values = b.Cont(b.Permute(v, 1, 2, 0, 3));                                      // [keys, width, heads]
        NetValue attended = b.MatMul(values, probabilities);                                     // [width, queries, heads]
        NetValue context = b.Reshape(b.Cont(b.Permute(attended, 0, 2, 1, 3)), _hidden, time);
        // Query rows past the valid frames attend nothing (the managed path leaves their context zero).
        return b.Linear(b.Mul(context, rowMask), prefix + ".o_proj.weight", Bias(prefix + ".o_proj"));
    }

    private NetValue ConvModule(NetworkBuilder b, NetValue x, NetValue rowMask, int time, string prefix)
    {
        NetValue up = b.Linear(x, b.Weight(prefix + ".pointwise_conv1.weight", _hidden, 2L * _hidden),
            Bias(prefix + ".pointwise_conv1") is { } upBias ? b.Vector(upBias, 2L * _hidden) : default);
        NetValue gated = b.Mul(b.Mul(b.Columns(up, 0, _hidden), b.Sigmoid(b.Columns(up, _hidden, _hidden))), rowMask);
        // 'same' zero padding of an odd kernel K: a causal conv over the rows with K / 2 zeros behind them, read K / 2 rows on.
        int half = _kernel / 2;
        NetValue padded = b.Concat(gated, b.Zeros(_hidden, half), 1);
        NetValue conv = b.CausalConv1d(padded, b.Weight(prefix + ".depthwise_conv.weight", _kernel, _hidden), default, out _);
        conv = b.Slice(conv, 1, half, time);
        if (Bias(prefix + ".depthwise_conv") is { } convBias) conv = b.Add(conv, b.Vector(convBias, _hidden));
        NetValue normed = b.Add(b.Mul(conv, b.Vector(prefix + ".norm.#scale", _hidden)), b.Vector(prefix + ".norm.#shift", _hidden));
        return b.Linear(b.Silu(normed), b.Weight(prefix + ".pointwise_conv2.weight", _hidden, _hidden),
            Bias(prefix + ".pointwise_conv2") is { } downBias ? b.Vector(downBias, _hidden) : default);
    }

    /// <summary><paramref name="prefix"/>.bias where the companion has it, else null.</summary>
    private string Bias(string prefix) => _graphWeights.Contains(prefix + ".bias") ? prefix + ".bias" : null;

    /// <summary>The sinusoid table of relative positions T - 1 .. -(T - 1), [2T - 1][hidden] (sin, cos pairs).</summary>
    private float[] RelativePositions(int time)
    {
        float[] positions = new float[checked((2 * time - 1) * _hidden)];
        for (int t = 0; t < 2 * time - 1; t++)
            for (int h = 0; h < _hidden; h += 2)
            {
                float angle = (time - 1 - t) * (1 / MathF.Pow(10000, (float)h / _hidden));
                positions[t * _hidden + h] = MathF.Sin(angle);
                positions[t * _hidden + h + 1] = MathF.Cos(angle);
            }
        return positions;
    }

    private static float[] TimeMask(int time, int valid)
    {
        var mask = new float[time];
        Array.Fill(mask, 1f, 0, Math.Min(valid, time));
        return mask;
    }

    /// <summary>F16 [keys, queries]: key s open below <paramref name="valid"/>, for every query.</summary>
    private static ushort[] KeyMask(int time, int valid)
    {
        const ushort open = 0, closed = 0xFC00;
        var mask = new ushort[(long)time * time];
        for (int t = 0; t < time; t++)
            for (int s = 0; s < time; s++)
                mask[(long)t * time + s] = s < valid ? open : closed;
        return mask;
    }

    private void DisposeGraph()
    {
        _graph?.Dispose();
        _graph = null;
        _graphWeights?.Dispose();
        _graphWeights = null;
    }
}
