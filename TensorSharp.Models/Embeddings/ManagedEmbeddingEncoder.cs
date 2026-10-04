// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Numerics.Tensors;
using System.Threading;

namespace TensorSharp.Models.Embeddings;

/// <summary>
/// Pure C# bidirectional encoder. Quantized weights stay quantized; projections
/// share a compact token batch, and attention is isolated by
/// sequence. No native dispatcher, tensor allocator or backend is initialized.
/// </summary>
internal sealed class ManagedEmbeddingEncoder : IDisposable
{
    private sealed class Matrix
    {
        public byte[] Data { get; private set; }
        private ManagedEmbeddingQ8Matrix _packed;
        public readonly GgmlTensorType Type;
        public readonly int Input, Output, RowBytes;
        public readonly float[] Bias;
        public Matrix(byte[] data, GgmlTensorType type, int input, int output, float[] bias = null)
        {
            Data = data; Type = type; Input = input; Output = output; Bias = bias;
            RowBytes = checked((int)ManagedQuantizedOps.RowSize((int)type, input));
        }
        public void Row(int row, float[] output, int offset) =>
            ManagedQuantizedOps.DequantizeToFloat32((int)Type, Data, checked(row * RowBytes), output, offset, Input);

        public void PrepareProjection()
        {
            if (Type != GgmlTensorType.Q8_0 || !ManagedEmbeddingQ8Matrix.IsSupported) return;
            _packed = new ManagedEmbeddingQ8Matrix(Data, Input, Output);
            Data = null;
        }

        public unsafe void Multiply(float[] input, int rows, float[] output, CpuWorkerPool pool, CancellationToken cancellationToken)
        {
            if (_packed != null) _packed.Multiply(input, rows, output, pool, cancellationToken);
            else
            fixed (byte* w = Data)
            fixed (float* x = input)
            fixed (float* y = output)
            {
                nint weights = (nint)w, inputs = (nint)x, outputs = (nint)y;
                int blocks = Math.Min(Output, Math.Max(1, pool.ThreadCount * 4));
                if (ManagedQuantizedOps.TryGetActivationPlan(Type, Input, out int activationRowBytes))
                {
                    var quantized = ArrayPool<byte>.Shared.Rent(checked(rows * activationRowBytes));
                    try
                    {
                        fixed (byte* q = quantized)
                        {
                            nint activations = (nint)q;
                            pool.For(rows, row => ManagedQuantizedOps.QuantizeActivationRow(Type,
                                (float*)inputs + (long)row * Input, (byte*)activations + (long)row * activationRowBytes, Input), cancellationToken);
                            pool.For(blocks, block =>
                            {
                                int first = Output * block / blocks, end = Output * (block + 1) / blocks;
                                for (int column = first; column < end; ++column)
                                for (int row = 0; row < rows; ++row)
                                    ((float*)outputs)[(long)row * Output + column] = ManagedQuantizedOps.DotQuantizedRow(Type,
                                        (byte*)weights + (long)column * RowBytes,
                                        (byte*)activations + (long)row * activationRowBytes, Input);
                            }, cancellationToken);
                        }
                    }
                    finally { ArrayPool<byte>.Shared.Return(quantized); }
                }
                else
                {
                    // Use the same owned scheduler for every format; this path
                    // never initializes the shared pool or native dispatcher.
                    pool.For(blocks, block =>
                    {
                        var scratch = ArrayPool<float>.Shared.Rent(Input);
                        try
                        {
                            fixed (float* rowWeights = scratch)
                            {
                                int first = Output * block / blocks, end = Output * (block + 1) / blocks;
                                for (int column = first; column < end; ++column)
                                {
                                    ManagedQuantizedOps.DequantizeRowToFloat32((int)Type,
                                        (IntPtr)((byte*)weights + (long)column * RowBytes), rowWeights, Input);
                                    int row = 0;
                                    for (; row + 4 <= rows; row += 4)
                                    {
                                        TensorComputePrimitives.Dot4((float*)inputs + (long)row * Input,
                                            (float*)inputs + (long)(row + 1) * Input,
                                            (float*)inputs + (long)(row + 2) * Input,
                                            (float*)inputs + (long)(row + 3) * Input,
                                            rowWeights, Input, out float a, out float b, out float c, out float d);
                                        ((float*)outputs)[(long)row * Output + column] = a;
                                        ((float*)outputs)[(long)(row + 1) * Output + column] = b;
                                        ((float*)outputs)[(long)(row + 2) * Output + column] = c;
                                        ((float*)outputs)[(long)(row + 3) * Output + column] = d;
                                    }
                                    for (; row < rows; ++row)
                                        ((float*)outputs)[(long)row * Output + column] =
                                            TensorComputePrimitives.Dot((float*)inputs + (long)row * Input, rowWeights, Input);
                                }
                            }
                        }
                        finally { ArrayPool<float>.Shared.Return(scratch); }
                    }, cancellationToken);
                }
            }
            if (Bias != null)
                for (int row = 0; row < rows; ++row)
                    TensorPrimitives.Add(output.AsSpan(row * Output, Output), Bias, output.AsSpan(row * Output, Output));
        }

    }

    private sealed record Norm(float[] Weight, float[] Bias);
    private sealed record Layer(Matrix Qkv, Matrix Q, Matrix K, Matrix V, Matrix Attention, Matrix Up, Matrix Gate, Matrix Down, Norm AttentionNorm, Norm OutputNorm);
    private readonly int _dim, _heads, _head, _ff, _pooling;
    private readonly bool _packedValues, _usesRoPE;
    private readonly float _epsilon;
    private readonly CpuWorkerPool _pool;
    private CancellationToken _cancellationToken;
    private Matrix _tokens, _positions;
    private float[] _type;
    private Norm _embeddingNorm;
    private Layer[] _layers;
    // Reuse activation storage across calls; the facade serializes this encoder.
    private float[] _x = [], _next = [], _qkv = [], _attention = [], _ffn = [], _projected = [], _transposedValues = [], _transposedKeys = [];

    public ManagedEmbeddingEncoder(GgufFile file, int threads)
    {
        bool isNomicBert = file.GetString("general.architecture") == "nomic-bert";
        _usesRoPE = isNomicBert;
        string prefix = isNomicBert ? "nomic-bert." : "bert.";
        _dim = Positive(file, prefix + "embedding_length");
        _heads = Positive(file, prefix + "attention.head_count");
        int ffRaw = Positive(file, prefix + "feed_forward_length");
        int context = Positive(file, prefix + "context_length");
        int layers = Positive(file, prefix + "block_count");
        _ff = isNomicBert ? _dim : ffRaw;
        if (_dim > 16384 || _dim % _heads != 0 || layers > 128 || ffRaw > 65536 || context > 65536)
            throw new InvalidDataException("Invalid embedding dimensions: maximum hidden size 16384, layers 128, feed-forward size 65536, and context 65536; hidden size must be divisible by head count.");
        _head = _dim / _heads;
        _packedValues = _head % ManagedEmbeddingMath.ValueWidth == 0;
        _epsilon = file.GetFloat32(prefix + "attention.layer_norm_epsilon", 1e-12f);
        if (!float.IsFinite(_epsilon) || _epsilon <= 0) throw new InvalidDataException("Invalid embedding layer normalization epsilon.");
        _pooling = checked((int)file.GetUint32(prefix + "pooling_type"));
        _tokens = ReadMatrix(file, "token_embd.weight", _dim);
        _positions = file.Tensors.ContainsKey("position_embd.weight") ? ReadMatrix(file, "position_embd.weight", _dim, context) : null;
        _embeddingNorm = ReadNorm(file, "token_embd_norm");
        if (file.Tensors.ContainsKey("token_types.weight"))
        {
            var types = ReadMatrix(file, "token_types.weight", _dim);
            _type = new float[_dim]; types.Row(0, _type, 0);
        }
        _layers = new Layer[layers];
        for (int i = 0; i < layers; ++i)
        {
            string blk = $"blk.{i}.";
            Matrix qkv = null, q = null, k = null, v = null;
            if (file.Tensors.ContainsKey(blk + "attn_qkv.weight"))
                qkv = ReadProjection(file, blk + "attn_qkv", _dim, checked(3 * _dim));
            else
            {
                q = ReadProjection(file, blk + "attn_q", _dim, _dim);
                k = ReadProjection(file, blk + "attn_k", _dim, _dim);
                v = ReadProjection(file, blk + "attn_v", _dim, _dim);
                if (q.Type == k.Type && q.Type == v.Type)
                {
                    var bytes = new byte[checked(q.Data.Length * 3)];
                    q.Data.CopyTo(bytes, 0); k.Data.CopyTo(bytes, q.Data.Length); v.Data.CopyTo(bytes, q.Data.Length * 2);
                    float[] bias = null;
                    if (q.Bias != null || k.Bias != null || v.Bias != null)
                    {
                        bias = new float[checked(3 * _dim)];
                        q.Bias?.CopyTo(bias, 0); k.Bias?.CopyTo(bias, _dim); v.Bias?.CopyTo(bias, 2 * _dim);
                    }
                    qkv = new Matrix(bytes, q.Type, _dim, checked(3 * _dim), bias);
                    q = k = v = null;
                }
            }
            Matrix up = ReadProjection(file, blk + "ffn_up", _dim, isNomicBert ? _dim : _ff);
            Matrix gate = isNomicBert ? ReadProjection(file, blk + "ffn_gate", _dim, _dim) : null;
            Matrix down = ReadProjection(file, blk + "ffn_down", isNomicBert ? _dim : _ff, _dim);
            _layers[i] = new Layer(qkv, q, k, v,
                ReadProjection(file, blk + "attn_output", _dim, _dim),
                up, gate, down,
                ReadNorm(file, blk + "attn_output_norm"), ReadNorm(file, blk + "layer_output_norm"));
            foreach (var projection in new[] { _layers[i].Qkv, _layers[i].Q, _layers[i].K, _layers[i].V,
                _layers[i].Attention, _layers[i].Up, _layers[i].Gate, _layers[i].Down }) projection?.PrepareProjection();
        }
        // Start owned workers only after all model validation/loading succeeds.
        _pool = new CpuWorkerPool(threads == 0 ? 4 : threads);
    }

    private static int Positive(GgufFile file, string key)
    {
        int value = checked((int)file.GetUint32(key));
        return value > 0 ? value : throw new InvalidDataException($"Invalid or missing embedding metadata '{key}'.");
    }
    private static Matrix ReadMatrix(GgufFile file, string name, int input, int output = -1)
    {
        if (!file.Tensors.TryGetValue(name, out var info)) throw new InvalidDataException($"Missing embedding tensor '{name}'.");
        if (info.Shape.Length is < 1 or > 2 || info.Shape[0] != (ulong)input ||
            (info.Shape.Length == 2 && (info.Shape[1] == 0 || info.Shape[1] > int.MaxValue)) ||
            (output >= 0 && (info.Shape.Length == 1 ? 1UL : info.Shape[1]) != (ulong)output))
            throw new InvalidDataException($"Embedding encoder: invalid dimensions for {name}.");
        if (!ManagedQuantizedOps.SupportsDequantization(info.Type))
            throw new NotSupportedException($"Pure C# embedding encoder does not support GGUF tensor type {info.Type} in '{name}'.");
        int count = info.Shape.Length == 1 ? 1 : (int)info.Shape[1];
        _ = checked((int)(ManagedQuantizedOps.RowSize((int)info.Type, input) * count));
        return new Matrix(file.ReadTensorData(info), info.Type, input, count);
    }
    private float[] ReadVector(GgufFile file, string name, int size)
    {
        var matrix = ReadMatrix(file, name, size, 1);
        var result = new float[size]; matrix.Row(0, result, 0);
        foreach (float value in result)
            if (!float.IsFinite(value)) throw new InvalidDataException($"Non-finite embedding parameter in '{name}'.");
        return result;
    }
    private Matrix ReadProjection(GgufFile file, string name, int input, int output)
    {
        var matrix = ReadMatrix(file, name + ".weight", input, output);
        var bias = file.Tensors.ContainsKey(name + ".bias") ? ReadVector(file, name + ".bias", output) : null;
        return new Matrix(matrix.Data, matrix.Type, input, output, bias);
    }
    private Norm ReadNorm(GgufFile file, string name) => new(ReadVector(file, name + ".weight", _dim), ReadVector(file, name + ".bias", _dim));

    public void Encode(int[] tokens, int[] lengths, float[] output, CancellationToken cancellationToken)
    {
        _cancellationToken = cancellationToken;
        int count = tokens.Length;
        Ensure(ref _x, checked(count * _dim)); Ensure(ref _next, checked(count * _dim));
        Ensure(ref _qkv, checked(count * _dim * 3)); Ensure(ref _attention, checked(count * _dim));
        Ensure(ref _ffn, checked(count * _ff)); Ensure(ref _projected, checked(count * _dim));
        var starts = new int[lengths.Length];
        var positions = new int[count];
        int queryGroupCount = 0;
        foreach (int length in lengths) queryGroupCount += (length + 3) / 4;
        var queryRows = new int[queryGroupCount];
        var querySequences = new int[queryGroupCount];
        int queryGroup = 0;
        int offset = 0;
        for (int s = 0; s < lengths.Length; ++s)
        {
            starts[s] = offset;
            for (int t = 0; t < lengths[s]; t += 4)
            {
                queryRows[queryGroup] = offset + t;
                querySequences[queryGroup++] = s;
            }
            for (int t = 0; t < lengths[s]; ++t)
            {
                int row = offset + t;
                positions[row] = t;
            }
            offset += lengths[s];
        }
        ForRows(count, row =>
        {
            _tokens.Row(tokens[row], _x, row * _dim);
            var x = _x.AsSpan(row * _dim, _dim);
            if (_positions != null)
            {
                _positions.Row(positions[row], _next, row * _dim);
                TensorPrimitives.Add(x, _next.AsSpan(row * _dim, _dim), x);
            }
            if (_type != null) TensorPrimitives.Add(x, _type, x);
        });
        NormalizeRows(_x, count, _embeddingNorm);
        for (int layerIndex = 0; layerIndex < _layers.Length; ++layerIndex)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var layer = _layers[layerIndex];
            if (layer.Qkv != null) layer.Qkv.Multiply(_x, count, _qkv, _pool, cancellationToken);
            else
            {
                layer.Q.Multiply(_x, count, _next, _pool, cancellationToken);
                layer.K.Multiply(_x, count, _attention, _pool, cancellationToken);
                layer.V.Multiply(_x, count, _projected, _pool, cancellationToken);
                for (int row = 0; row < count; ++row)
                {
                    Array.Copy(_next, row * _dim, _qkv, row * _dim * 3, _dim);
                    Array.Copy(_attention, row * _dim, _qkv, row * _dim * 3 + _dim, _dim);
                    Array.Copy(_projected, row * _dim, _qkv, row * _dim * 3 + 2 * _dim, _dim);
                }
            }
            if (_usesRoPE) RotateQkv(count, positions);
            bool select = layerIndex == _layers.Length - 1 && _pooling != 1;
            int rows = select ? lengths.Length : count;
            Attention(lengths, starts, queryRows, querySequences, count, select, cancellationToken);
            layer.Attention.Multiply(_attention, rows, _next, _pool, cancellationToken);
            for (int row = 0; row < rows; ++row)
            {
                int source = select ? starts[row] + (_pooling == 3 ? lengths[row] - 1 : 0) : row;
                TensorPrimitives.Add(_next.AsSpan(row * _dim, _dim), _x.AsSpan(source * _dim, _dim), _next.AsSpan(row * _dim, _dim));
            }
            NormalizeRows(_next, rows, layer.AttentionNorm);
            if (layer.Gate != null)
            {
                layer.Up.Multiply(_next, rows, _ffn, _pool, cancellationToken);
                layer.Gate.Multiply(_next, rows, _projected, _pool, cancellationToken);
                SwiGlu(rows);
            }
            else
            {
                layer.Up.Multiply(_next, rows, _ffn, _pool, cancellationToken);
                Gelu(rows);
            }
            layer.Down.Multiply(_ffn, rows, _x, _pool, cancellationToken);
            TensorPrimitives.Add(_x.AsSpan(0, rows * _dim), _next.AsSpan(0, rows * _dim), _x.AsSpan(0, rows * _dim));
            NormalizeRows(_x, rows, layer.OutputNorm);
        }
        for (int s = 0; s < lengths.Length; ++s)
        {
            var result = output.AsSpan(s * _dim, _dim);
            if (_pooling == 1)
            {
                result.Clear();
                for (int row = starts[s]; row < starts[s] + lengths[s]; ++row)
                    TensorPrimitives.Add(result, _x.AsSpan(row * _dim, _dim), result);
                TensorPrimitives.Multiply(result, 1.0f / lengths[s], result);
            }
            else _x.AsSpan(s * _dim, _dim).CopyTo(result);
            double sum = 0;
            foreach (float value in result) sum += (double)value * value;
            if (!double.IsFinite(sum) || sum <= 0) throw new InvalidOperationException("Embedding encoder produced a non-finite or zero vector.");
            double scale = 1.0 / Math.Sqrt(sum);
            for (int i = 0; i < result.Length; ++i) result[i] = (float)(result[i] * scale);
        }
        cancellationToken.ThrowIfCancellationRequested();
    }

    private unsafe void Attention(int[] lengths, int[] starts, int[] queryRows, int[] querySequences,
        int count, bool select, CancellationToken cancellationToken)
    {
        Ensure(ref _transposedValues, checked(count * _dim));
        int keyStride = ManagedEmbeddingMath.PaddedKeyStride(count);
        Ensure(ref _transposedKeys, checked(keyStride * _dim));
        // Store key/value channels contiguously across tokens. Score lanes hold
        // adjacent keys, and long value reductions reuse each channel for four
        // queries without short per-key vector updates.
        fixed (float* qkvPointer = _qkv)
        fixed (float* valuePointer = _transposedValues)
        fixed (float* keyPointer = _transposedKeys)
        fixed (float* attentionPointer = _attention)
        {
            nint qkvAddress = (nint)qkvPointer, valueAddress = (nint)valuePointer, keyAddress = (nint)keyPointer, attentionAddress = (nint)attentionPointer;
            const int tileDimensions = 16, tileTokens = 32;
            _pool.For((_dim + tileDimensions - 1) / tileDimensions, tile =>
            {
                int firstDim = tile * tileDimensions, endDim = Math.Min(firstDim + tileDimensions, _dim);
                for (int firstToken = 0; firstToken < count; firstToken += tileTokens)
                {
                    int endToken = Math.Min(firstToken + tileTokens, count);
                    for (int dim = firstDim; dim < endDim; ++dim)
                    for (int token = firstToken; token < endToken; ++token)
                    {
                        if (_packedValues)
                        {
                            int width = ManagedEmbeddingMath.ValueWidth;
                            if (dim % width == 0)
                                TensorComputePrimitives.StoreVector((float*)valueAddress + (long)dim * count + token * width,
                                    TensorComputePrimitives.LoadVector((float*)qkvAddress + (long)token * 3 * _dim + 2 * _dim + dim));
                        }
                        else
                            ((float*)valueAddress)[(long)dim * count + token] = ((float*)qkvAddress)[(long)token * 3 * _dim + 2 * _dim + dim];
                        ((float*)keyAddress)[(long)dim * keyStride + token] = ((float*)qkvAddress)[(long)token * 3 * _dim + _dim + dim];
                    }
                }
            }, cancellationToken);
            int maximum = 0;
            foreach (int length in lengths) maximum = Math.Max(maximum, length);
            if (maximum >= 1024 && _packedValues && !select)
            {
                const int tile = ManagedEmbeddingLongAttention.QueryTile;
                int tileCount = 0;
                foreach (int length in lengths) tileCount += (length + tile - 1) / tile;
                var tileRows = new int[tileCount];
                var tileSequences = new int[tileCount];
                int nextTile = 0;
                for (int sequence = 0; sequence < lengths.Length; ++sequence)
                for (int row = 0; row < lengths[sequence]; row += tile)
                {
                    tileRows[nextTile] = starts[sequence] + row;
                    tileSequences[nextTile++] = sequence;
                }
                ForRowsWithScratch(checked(tileCount * _heads), ManagedEmbeddingLongAttention.ScratchSize(_head), (index, scratch) =>
                {
                    int head = index / tileCount, group = index % tileCount;
                    int sequence = tileSequences[group], first = starts[sequence], length = lengths[sequence];
                    int firstQuery = tileRows[group], queries = Math.Min(tile, first + length - firstQuery);
                    ManagedEmbeddingLongAttention.Compute(
                        (float*)qkvAddress + (long)firstQuery * 3 * _dim + head * _head, 3 * _dim,
                        (float*)keyAddress + (long)head * _head * keyStride + first, keyStride,
                        (float*)valueAddress + (long)head * _head * count + first * ManagedEmbeddingMath.ValueWidth, count,
                        _head, length, queries, (float*)attentionAddress + (long)firstQuery * _dim + head * _head,
                        _dim, scratch, cancellationToken);
                });
                return;
            }
            float scale = 1.0f / MathF.Sqrt(_head);
            int groups = select ? lengths.Length : queryRows.Length;
            // Long sequences exceed cache capacity when successive jobs cycle
            // through every head's keys and values. Keep a head's independent
            // query groups adjacent so its K/V data stays hot across jobs.
            bool headMajor = maximum >= 1024;
            ForRowsWithScratch(checked(groups * _heads), checked(maximum * 4), (index, scratch) =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                int group = headMajor ? index % groups : index / _heads;
                int head = headMajor ? index / groups : index % _heads;
                int sequence = select ? group : querySequences[group];
                int first = starts[sequence], length = lengths[sequence];
                int firstQuery = select ? first + (_pooling == 3 ? length - 1 : 0) : queryRows[group];
                int queries = select ? 1 : Math.Min(4, first + length - firstQuery);
                int q1 = Math.Min(1, queries - 1), q2 = Math.Min(2, queries - 1), q3 = Math.Min(3, queries - 1);
                float* query = (float*)qkvAddress + (long)firstQuery * 3 * _dim + head * _head;
                int queryStride = 3 * _dim;
                fixed (float* scores = scratch)
                {
                    ManagedEmbeddingMath.Scores(query, query + q1 * queryStride, query + q2 * queryStride, query + q3 * queryStride,
                        (float*)keyAddress + (long)head * _head * keyStride + first, keyStride, _head, length, scale, scores);
                    // Duplicate tail query slots intentionally: every Dot4 lane
                    // is initialized, while stores below only touch real rows.
                    for (int q = 0; q < 4; ++q)
                    {
                        var probabilities = new Span<float>(scores + q * length, length);
                        float maximumScore = TensorPrimitives.Max(probabilities);
                        TensorPrimitives.Subtract(probabilities, maximumScore, probabilities);
                        TensorPrimitives.Exp(probabilities, probabilities);
                        TensorPrimitives.Multiply(probabilities, 1.0f / TensorPrimitives.Sum(probabilities), probabilities);
                    }
                    int destination = select ? sequence : firstQuery;
                    if (_packedValues)
                    {
                        ManagedEmbeddingMath.Values(scores, length,
                            (float*)valueAddress + (long)head * _head * count + first * ManagedEmbeddingMath.ValueWidth,
                            count, _head, (float*)attentionAddress + (long)destination * _dim + head * _head, _dim, queries);
                        return;
                    }
                    for (int dim = 0; dim < _head; ++dim)
                    {
                        ManagedEmbeddingMath.Dot4(scores, scores + length, scores + 2 * length, scores + 3 * length,
                            (float*)valueAddress + (long)(head * _head + dim) * count + first,
                            length, out float a0, out float a1, out float a2, out float a3);
                        float* output = (float*)attentionAddress + (long)destination * _dim + head * _head + dim;
                        output[0] = a0;
                        if (queries > 1) output[_dim] = a1;
                        if (queries > 2) output[2 * _dim] = a2;
                        if (queries > 3) output[3 * _dim] = a3;
                    }
                }
            });
        }
    }

    private void NormalizeRows(float[] values, int rows, Norm norm)
    {
        ForRows(rows, row =>
        {
            var x = values.AsSpan(row * _dim, _dim);
            float mean = TensorPrimitives.Sum(x) / _dim;
            TensorPrimitives.Subtract(x, mean, x);
            float variance = TensorPrimitives.Dot(x, x) / _dim;
            TensorPrimitives.Multiply(x, 1.0f / MathF.Sqrt(variance + _epsilon), x);
            TensorPrimitives.Multiply(x, norm.Weight, x);
            TensorPrimitives.Add(x, norm.Bias, x);
        });
    }

    private void Gelu(int rows)
    {
        ForRowsWithScratch(rows, _ff, (row, scratch) =>
        {
            var x = _ffn.AsSpan(row * _ff, _ff);
            var y = scratch.AsSpan(0, _ff);
            TensorPrimitives.Multiply(x, x, y);
            TensorPrimitives.Multiply(y, x, y);
            TensorPrimitives.MultiplyAdd(y, 0.044715f, x, y);
            TensorPrimitives.Multiply(y, 0.7978845608028654f, y);
            TensorPrimitives.Tanh(y, y);
            TensorPrimitives.Add(y, 1.0f, y);
            TensorPrimitives.Multiply(x, y, x);
            TensorPrimitives.Multiply(x, 0.5f, x);
        });
    }

    private void SwiGlu(int rows)
    {
        // SwiGLU: ffn = silu(gate) * up
        // silu(x) = x * sigmoid(x).
        // gate is in _projected, up is in _ffn; result stays in _ffn.
        ForRows(rows, row =>
        {
            var gate = _projected.AsSpan(row * _ff, _ff);
            var up = _ffn.AsSpan(row * _ff, _ff);
            for (int i = 0; i < _ff; ++i)
            {
                float g = gate[i];
                float sigmoid = 1.0f / (1.0f + MathF.Exp(-g));
                up[i] *= g * sigmoid;
            }
        });
    }

    private void RotateQkv(int count, int[] positions)
    {
        // Apply RoPE to Q and K portions of _qkv in-place.
        // _qkv layout: [token * 3 * _dim] where [0:_dim]=Q, [_dim:2*_dim]=K, [2*_dim:3*_dim]=V.
        // Each head's Q/K has _head dimensions; RoPE rotates pairs (2j, 2j+1) by pos * theta_base^(-2j/_head).
        int halfHead = _head / 2;
        float[] qkv = _qkv;
        ForRows(count, row =>
        {
            int pos = positions[row];
            var qRow = qkv.AsSpan(row * 3 * _dim, _dim);
            var kRow = qkv.AsSpan(row * 3 * _dim + _dim, _dim);
            for (int h = 0; h < _heads; ++h)
            {
                var qHead = qRow.Slice(h * _head, _head);
                var kHead = kRow.Slice(h * _head, _head);
                for (int j = 0; j < halfHead; ++j)
                {
                    float theta = pos * MathF.Pow(1000.0f, -2.0f * j / _head);
                    float cos = MathF.Cos(theta), sin = MathF.Sin(theta);
                    float q0 = qHead[2 * j], q1 = qHead[2 * j + 1];
                    qHead[2 * j] = q0 * cos - q1 * sin;
                    qHead[2 * j + 1] = q1 * cos + q0 * sin;
                    float k0 = kHead[2 * j], k1 = kHead[2 * j + 1];
                    kHead[2 * j] = k0 * cos - k1 * sin;
                    kHead[2 * j + 1] = k1 * cos + k0 * sin;
                }
            }
        });
    }

    private void ForRows(int rows, Action<int> action)
    {
        int tasks = Math.Min(rows, _pool.ThreadCount * 4);
        _pool.For(tasks, task =>
        {
            int end = (int)((long)rows * (task + 1) / tasks);
            for (int row = (int)((long)rows * task / tasks); row < end; ++row) action(row);
        }, _cancellationToken);
    }

    private void ForRowsWithScratch(int rows, int scratchSize, Action<int, float[]> action)
    {
        int tasks = Math.Min(rows, _pool.ThreadCount * 4);
        _pool.For(tasks, task =>
        {
            var scratch = ArrayPool<float>.Shared.Rent(scratchSize);
            try
            {
                int end = (int)((long)rows * (task + 1) / tasks);
                for (int row = (int)((long)rows * task / tasks); row < end; ++row) action(row, scratch);
            }
            finally { ArrayPool<float>.Shared.Return(scratch); }
        }, _cancellationToken);
    }

    private static void Ensure(ref float[] buffer, int size)
    {
        if (buffer.Length >= size) return;
        var replacement = ArrayPool<float>.Shared.Rent(size);
        if (buffer.Length != 0) ArrayPool<float>.Shared.Return(buffer);
        buffer = replacement;
    }
    public void Dispose()
    {
        _pool.Dispose();
        foreach (var buffer in new[] { _x, _next, _qkv, _attention, _ffn, _projected, _transposedValues, _transposedKeys })
            if (buffer.Length != 0) ArrayPool<float>.Shared.Return(buffer);
        _x = _next = _qkv = _attention = _ffn = _projected = [];
        _transposedValues = _transposedKeys = [];
        _tokens = _positions = null; _type = null; _embeddingNorm = null; _layers = [];
    }
}
