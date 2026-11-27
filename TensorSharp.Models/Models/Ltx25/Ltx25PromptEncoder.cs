using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using TensorSharp.Cuda;
using TensorSharp.Models.Direct;
using TensorSharp.Models.Video;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Ltx25;

public sealed record Ltx25PromptEncoding(float[] VideoContext, float[] AudioContext, int Tokens,
    int[] TokenIds, int[] AttentionMask);

/// <summary>Native Gemma4 Unified text conditioning, BF16 hidden-state aggregation,
/// and the checkpoint's learned video/audio embedding connectors.</summary>
public sealed class Ltx25PromptEncoder : IDisposable
{
    private readonly SafetensorsFile _file;
    private readonly DirectContext _context;
    private readonly Ltx25PromptConnector _video, _audio;
    private readonly Configuration _config;
    private readonly bool _cudaBf16Supported;
    private readonly Ltx25CudaFilePrefault _cudaPrefault;
    private MappedFileReadAhead _readAhead;
    private bool _readAheadWarned;
    private readonly bool _releaseTextCache;
    // Dropping 22 GiB of cached pages took 1.5 s on an A40 pod's network volume; it
    // runs beside the denoiser and the next encode waits for it before reading again.
    private System.Threading.Tasks.Task _pendingRelease;
    private readonly string _releasePolicy;
    private bool _disposed;
    public Ltx25PromptTokenizer Tokenizer { get; }
    public double LastGemmaMilliseconds { get; private set; }
    public double LastAggregationMilliseconds { get; private set; }
    public double LastConnectorMilliseconds { get; private set; }
    /// <summary>The last encode's weight read-ahead (and whether its cache is being released), or null when none ran.</summary>
    public string LastReadAhead { get; private set; }
    public long CudaMappedProjections { get; private set; }
    public long CudaBf16Projections { get; private set; }
    public long CudaPrefaultReadBytes => _cudaPrefault?.ReadBytes ?? 0;
    public long CudaPrefaultResidentBytes => _cudaPrefault?.ResidentBytes ?? 0;
    public long CudaPrefaultRanges => _cudaPrefault?.Ranges ?? 0;
    public double CudaPrefaultMilliseconds => _cudaPrefault?.Milliseconds ?? 0;
    public string CudaPrefaultFailure => _cudaPrefault?.Failure;
    public string CudaModulePath => _video.CudaModulePath;
    public long CudaConnectorProjections => _video.CudaProjections + _audio.CudaProjections;
    public long CudaConnectorAttentions => _video.CudaAttentions + _audio.CudaAttentions;
    private void Count(Ltx25MappedLinear linear)
    {
        if (linear.UsedCudaMapping) ++CudaMappedProjections;
        if (linear.UsedCudaBf16) ++CudaBf16Projections;
    }
    internal Action<string, float[]> Trace { get; set; }
    private void Capture(string name, Tensor value) => Trace?.Invoke(name, value.GetElementsAsFloat(checked((int)value.ElementCount())));

    public Ltx25PromptEncoder(string checkpoint, string denoiserPath, IAllocator allocator)
    {
        ArgumentNullException.ThrowIfNull(allocator);
        _file = new(checkpoint);
        try
        {
            _config = Configuration.Read(_file);
            Tokenizer = new(_file.ReadRawBytes("tokenizer_json"));
            _context = new(allocator);
            // Native BF16 GEMM requires Ampere or newer; retain mapped F32
            // projection on older CUDA devices even when the candidate is requested.
            _cudaBf16Supported = allocator is CudaAllocator cuda &&
                CudaDevice.GetDevice(cuda.DeviceId).ComputeCapabilityMajor >= 8;
            if (_context.IsCuda) _cudaPrefault = new(_file);
            // A text-encoder file cache left behind after encoding competes with the
            // denoiser: 22 GiB of stale Gemma pages added ~10 s to the first 1024x1024
            // denoise step on a 48 GiB M5 Pro, and in a memory-limited container they
            // evict the denoiser file the upload is about to read. Release it when the
            // machine (or container) cannot also hold the denoiser file with headroom;
            // the read-ahead makes a later re-read cheap.
            long physical = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
            long needed = checked(new FileInfo(checkpoint).Length + new FileInfo(denoiserPath).Length + (16L << 30));
            _releaseTextCache = physical > 0 && physical < needed;
            _releasePolicy = $"{physical / (double)(1L << 30):F0} GiB RAM < {needed / (double)(1L << 30):F0} GiB " +
                "(text encoder + denoiser + 16 GiB headroom)";
            _video = new(denoiserPath, allocator, "video_embeddings_connector");
            _audio = new(denoiserPath, allocator, "audio_embeddings_connector");
            ValidateProjection("video", _video.Dimension);
            ValidateProjection("audio", _audio.Dimension);
        }
        catch { _audio?.Dispose(); _video?.Dispose(); _cudaPrefault?.Dispose(); _context?.Dispose(); _file.Dispose(); throw; }
    }

    private void ValidateProjection(string stream, int output)
    {
        var info = _file.GetInfo("text_embedding_projection." + stream + "_aggregate_embed.weight");
        if (!info.Shape.SequenceEqual(new long[] { output, checked(_config.Hidden * (_config.Layers + 1)) }))
            throw new InvalidDataException("LTX Gemma hidden-state projection dimensions differ from the connectors.");
    }

    public Ltx25PromptEncoding Encode(string prompt, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int[] ids = Tokenizer.Encode(prompt);
        _pendingRelease?.Wait();
        _pendingRelease = null;
        var readAhead = StartReadAhead();
        _readAhead = readAhead;
        try { return Encode(ids, cancellationToken); }
        finally
        {
            _readAhead = null;
            LastReadAhead = null;
            if (readAhead != null)
            {
                readAhead.Stop();
                LastReadAhead = readAhead.Describe();
                if (_releaseTextCache)
                {
                    LastReadAhead += $" releasing in the background ({_releasePolicy})";
                    _pendingRelease = System.Threading.Tasks.Task.Run(() => { try { readAhead.ReleaseCache(); } finally { readAhead.Dispose(); } });
                }
                else readAhead.Dispose();
            }
        }
    }

    // MLX and CPU read the mapped weights directly, so a cold checkpoint is
    // otherwise read through page faults: 24 s of a 28 s first encode on an M5 Pro.
    // CUDA warms each matrix before its upload (Ltx25CudaFilePrefault), one matrix
    // at a time; reading ahead with concurrent readers lets that warm find the
    // pages already cached.
    private MappedFileReadAhead StartReadAhead()
    {
        var stages = new List<(long Offset, long Length)>[_config.Layers + 1];
        for (int i = 0; i < stages.Length; ++i) stages[i] = new();
        const string layers = "model.layers.";
        foreach (var info in _file.Tensors.Values)
        {
            int stage;
            if (info.Name.StartsWith(layers, StringComparison.Ordinal))
            {
                int end = info.Name.IndexOf('.', layers.Length);
                if (end < 0 || !int.TryParse(info.Name.AsSpan(layers.Length, end - layers.Length), out stage) ||
                    (uint)stage >= (uint)_config.Layers) continue;
            }
            else if (info.Name == "model.norm.weight" || info.Name.StartsWith("text_embedding_projection.", StringComparison.Ordinal))
                stage = _config.Layers;
            else continue; // the token table is read row by row; vision and audio towers are unused
            stages[stage].Add((checked(_file.DataOffset + info.Begin), info.ByteCount));
        }
        return MappedFileReadAhead.TryStart(_file.Path, stages.Select(stage => stage.OrderBy(range => range.Offset)), message =>
        {
            if (!_readAheadWarned) Console.WriteLine("  [ltx25] warning: text-encoder " + message);
            _readAheadWarned = true;
        });
    }

    private Ltx25PromptEncoding Encode(int[] ids, CancellationToken cancellationToken)
    {
        var timer = Stopwatch.StartNew();
        var states = EncodeHiddenStates(ids, cancellationToken);
        LastGemmaMilliseconds = timer.Elapsed.TotalMilliseconds; timer.Restart();
        _readAhead?.Enter(_config.Layers);
        using var features = _context.FromFloats(Features(states, ids.Length), ids.Length,
            checked(_config.Hidden * (_config.Layers + 1)));
        float[] video = Aggregate(features, _video.Dimension, "video", cancellationToken);
        cancellationToken.ThrowIfCancellationRequested();
        float[] audio = Aggregate(features, _audio.Dimension, "audio", cancellationToken);
        LastAggregationMilliseconds = timer.Elapsed.TotalMilliseconds; timer.Restart();
        video = _video.Forward(video, ids.Length, 1024, cancellationToken);
        audio = _audio.Forward(audio, ids.Length, 1024, cancellationToken);
        LastConnectorMilliseconds = timer.Elapsed.TotalMilliseconds;
        var paddedIds = new int[1024]; Array.Copy(ids, paddedIds, ids.Length);
        var mask = Enumerable.Repeat(1, 1024).ToArray();
        return new(video, audio, 1024, paddedIds, mask);
    }

    // Hidden states are BF16 values widened to F32. Only valid tokens are run;
    // causal attention makes left padding irrelevant, but absolute RoPE positions
    // must retain the original 1024-token, left-padded positions.
    internal float[][] EncodeHiddenStates(int[] ids, CancellationToken cancellationToken = default)
    {
        int count = ids.Length, hidden = _config.Hidden;
        if (count is < 1 or > 1024) throw new ArgumentOutOfRangeException(nameof(ids));
        var embeddings = new float[checked(count * hidden)];
        var table = _file.GetInfo("model.embed_tokens.weight");
        for (int token = 0; token < count; ++token)
        {
            if ((uint)ids[token] >= table.Shape[0]) throw new ArgumentOutOfRangeException(nameof(ids));
            var row = ReadRows(table, ids[token], 1);
            float scale = Ltx25PromptMath.Bf16(MathF.Sqrt(hidden));
            for (int i = 0; i < hidden; ++i) embeddings[token * hidden + i] = Ltx25PromptMath.Bf16(row[i] * scale);
        }
        var states = new float[_config.Layers + 1][]; states[0] = embeddings;
        Tensor x = _context.FromFloats(embeddings, count, hidden);
        var causal = new float[checked(_config.Heads * count * count)];
        for (int h = 0; h < _config.Heads; ++h)
        for (int q = 0; q < count; ++q)
        for (int k = q + 1; k < count; ++k) causal[(h * count + q) * count + k] = float.NegativeInfinity;
        using var mask = _context.FromFloats(causal, _config.Heads, count, count);
        try
        {
            for (int layer = 0; layer < _config.Layers; ++layer)
            {
                cancellationToken.ThrowIfCancellationRequested();
                _readAhead?.Enter(layer);
                var next = Layer(x, mask, layer, 1024 - count, cancellationToken);
                x.Dispose(); x = next;
                if (layer + 1 < _config.Layers) states[layer + 1] = x.GetElementsAsFloat(checked((int)x.ElementCount()));
            }
            using var gain = Gain("model.norm.weight");
            using var normalized = Ltx25PromptMath.Norm(_context, x, gain, _config.Epsilon, true);
            states[^1] = normalized.GetElementsAsFloat(checked((int)normalized.ElementCount()));
            return states;
        }
        finally { x.Dispose(); }
    }

    private Tensor Gain(string name) => _context.FromFloats(_file.ReadFloat32(name), _file.GetInfo(name).NumElements);
    private Tensor Norm(Tensor x, string name)
    {
        Capture(name[..^7] + ".input", x);
        using var gain = Gain(name);
        var result = Ltx25PromptMath.Norm(_context, x, gain, _config.Epsilon, true);
        Capture(name[..^7] + ".output", result); return result;
    }
    private Tensor HeadNorm(Tensor x, string name, int dim)
    {
        using var rows = x.View(x.ElementCount() / dim, dim);
        using var gain = name == null ? null : Gain(name);
        using var normalized = Ltx25PromptMath.Norm(_context, rows, gain, _config.Epsilon, true);
        if (name != null) Capture(name[..^7] + ".output", normalized);
        return normalized.View(x.Sizes);
    }
    private Tensor Dense(Tensor x, string name, CancellationToken cancellationToken)
    {
        Capture(name + ".input", x);
        using var linear = new Ltx25MappedLinear(_file, _context, name, _cudaBf16Supported, _cudaPrefault);
        var result = linear.Forward(x, true, cancellationToken); Count(linear); Capture(name + ".output", result); return result;
    }

    private Tensor Layer(Tensor x, Tensor mask, int layer, int position, CancellationToken cancellationToken)
    {
        string p = "model.layers." + layer;
        bool global = _config.Types[layer] == "full_attention";
        int dim = global ? _config.GlobalHead : _config.Head, kv = global ? _config.GlobalKv : _config.Kv;
        using var input = Norm(x, p + ".input_layernorm.weight");
        using var rawq = Dense(input, p + ".self_attn.q_proj", cancellationToken);
        using var rawk = Dense(input, p + ".self_attn.k_proj", cancellationToken);
        using var rawv = global ? null : Dense(input, p + ".self_attn.v_proj", cancellationToken);
        using var q = HeadNorm(rawq, p + ".self_attn.q_norm.weight", dim);
        using var k = HeadNorm(rawk, p + ".self_attn.k_norm.weight", dim);
        using var v = HeadNorm(rawv ?? rawk, null, dim);
        Ltx25PromptMath.GemmaRope(q, _config.Heads, dim, position, global ? _config.GlobalTheta : _config.Theta,
            global ? _config.Partial : 1);
        Ltx25PromptMath.GemmaRope(k, kv, dim, position, global ? _config.GlobalTheta : _config.Theta,
            global ? _config.Partial : 1);
        using var repeatedK = Ltx25PromptMath.RepeatKv(_context, k, _config.Heads, kv, dim);
        using var repeatedV = Ltx25PromptMath.RepeatKv(_context, v, _config.Heads, kv, dim);
        using var attended = DirectOps.Attention(_context, q, repeatedK, repeatedV, _config.Heads, dim, 1, mask);
        Ltx25PromptMath.RoundBf16(attended);
        using var projected = Dense(attended, p + ".self_attn.o_proj", cancellationToken);
        using var attention = Norm(projected, p + ".post_attention_layernorm.weight");
        using var residual = _context.NewF32(x.Sizes); Ops.Add(residual, x, attention); Ltx25PromptMath.RoundBf16(residual);
        using var ffInput = Norm(residual, p + ".pre_feedforward_layernorm.weight");
        using var gate = Dense(ffInput, p + ".mlp.gate_proj", cancellationToken);
        using var up = Dense(ffInput, p + ".mlp.up_proj", cancellationToken);
        using var activated = DirectOps.Gelu(_context, gate); Ltx25PromptMath.RoundBf16(activated);
        using var product = _context.NewF32(up.Sizes); Ops.Mul(product, up, activated); Ltx25PromptMath.RoundBf16(product);
        using var down = Dense(product, p + ".mlp.down_proj", cancellationToken);
        using var ff = Norm(down, p + ".post_feedforward_layernorm.weight");
        var output = _context.NewF32(x.Sizes);
        Ops.Add(output, residual, ff); Ltx25PromptMath.RoundBf16(output);
        Ops.Mul(output, output, _file.ReadFloat32(p + ".layer_scalar")[0]); Ltx25PromptMath.RoundBf16(output);
        return output;
    }

    private float[] ReadRows(SafetensorTensorInfo info, int start, int rows)
    {
        int width = checked((int)info.Shape[1]);
        int bytes = info.Dtype == SafetensorDtype.F32 ? 4 : 2;
        long begin = info.Begin + (long)start * width * bytes;
        var slice = new SafetensorTensorInfo { Name = info.Name, Shape = [rows, width], Dtype = info.Dtype,
            Begin = begin, End = begin + (long)rows * width * bytes };
        var values = new float[checked(rows * width)]; _file.ReadTensorDataToFloat32(slice, values, values.Length);
        return values;
    }

    internal float[] Features(float[][] states, int tokens)
    {
        int hidden = _config.Hidden, layers = states.Length;
        var output = new float[checked(tokens * hidden * layers)];
        for (int token = 0; token < tokens; ++token)
        for (int layer = 0; layer < layers; ++layer)
        {
            float sum = 0;
            for (int d = 0; d < hidden; ++d)
            { float value = states[layer][token * hidden + d]; sum += Ltx25PromptMath.Bf16(value * value); }
            float variance = Ltx25PromptMath.Bf16(sum / hidden);
            float inverse = Ltx25PromptMath.Bf16(1 / MathF.Sqrt(Ltx25PromptMath.Bf16(variance + 1e-6f)));
            for (int d = 0; d < hidden; ++d)
                output[(token * hidden + d) * layers + layer] = Ltx25PromptMath.Bf16(states[layer][token * hidden + d] * inverse);
        }
        return output;
    }

    internal float[] Aggregate(Tensor features, int dimension, string stream, CancellationToken cancellationToken = default)
    {
        using var scaled = _context.NewF32(features.Sizes);
        Ops.Mul(scaled, features, MathF.Sqrt((float)dimension / _config.Hidden)); Ltx25PromptMath.RoundBf16(scaled);
        using var projection = new Ltx25MappedLinear(_file, _context, "text_embedding_projection." + stream + "_aggregate_embed", _cudaBf16Supported, _cudaPrefault);
        using var output = projection.Forward(scaled, true, cancellationToken);
        Count(projection);
        return output.GetElementsAsFloat(checked((int)output.ElementCount()));
    }

    public void ReleaseDeviceResidency() { ObjectDisposedException.ThrowIf(_disposed, this); _video.ReleaseDeviceResidency(); _audio.ReleaseDeviceResidency(); }
    public void Dispose() { if (_disposed) return; _disposed = true; _pendingRelease?.Wait(); _audio.Dispose(); _video.Dispose(); _cudaPrefault?.Dispose(); _context.Dispose(); _file.Dispose(); }

    private sealed record Configuration(int Hidden, int Layers, int Heads, int Kv, int GlobalKv, int Head,
        int GlobalHead, float Epsilon, string[] Types, float Theta, float GlobalTheta, float Partial)
    {
        internal static Configuration Read(SafetensorsFile file)
        {
            using var document = JsonDocument.Parse(file.Metadata["gemma_config"]);
            var root = document.RootElement;
            if (root.GetProperty("model_type").GetString() != "gemma4_unified") throw new NotSupportedException("LTX requires Gemma4 Unified.");
            var c = root.GetProperty("text_config");
            if (c.GetProperty("num_kv_shared_layers").GetInt32() != 0 || !c.GetProperty("attention_k_eq_v").GetBoolean() ||
                c.GetProperty("hidden_size_per_layer_input").GetInt32() != 0 || c.GetProperty("enable_moe_block").GetBoolean())
                throw new NotSupportedException("Unsupported Gemma4 Unified text graph configuration.");
            int hidden = c.GetProperty("hidden_size").GetInt32(), layers = c.GetProperty("num_hidden_layers").GetInt32();
            var types = c.GetProperty("layer_types").EnumerateArray().Select(v => v.GetString()).ToArray();
            var rope = c.GetProperty("rope_parameters");
            if (types.Length != layers || types.Any(t => t != "full_attention" && t != "sliding_attention") ||
                c.GetProperty("sliding_window").GetInt32() < 1024 || hidden <= 0 || layers <= 0)
                throw new NotSupportedException("Unsupported Gemma4 Unified attention geometry.");
            return new(hidden, layers, c.GetProperty("num_attention_heads").GetInt32(), c.GetProperty("num_key_value_heads").GetInt32(),
                c.GetProperty("num_global_key_value_heads").GetInt32(), c.GetProperty("head_dim").GetInt32(),
                c.GetProperty("global_head_dim").GetInt32(), c.GetProperty("rms_norm_eps").GetSingle(), types,
                rope.GetProperty("sliding_attention").GetProperty("rope_theta").GetSingle(),
                rope.GetProperty("full_attention").GetProperty("rope_theta").GetSingle(),
                rope.GetProperty("full_attention").GetProperty("partial_rotary_factor").GetSingle());
        }
    }
}
