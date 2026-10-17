// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.Direct;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Ltx25;

/// <summary>Batch-one, unmasked AV denoiser inputs. Contexts are already
/// projected to the two checkpoint stream widths. No text encoder or connector
/// is called here. Latents are token-major, not VAE channel-major storage.</summary>
public sealed class Ltx25DenoiserInput
{
    public required float[] Video { get; init; }
    public required int VideoTokens { get; init; }
    public required float[] Audio { get; init; }
    public required int AudioTokens { get; init; }
    public required float[] VideoContext { get; init; }
    public required int VideoContextTokens { get; init; }
    public required float[] AudioContext { get; init; }
    public required int AudioContextTokens { get; init; }
    public required Ltx25PositionGrid VideoPositions { get; init; }
    public required Ltx25PositionGrid AudioPositions { get; init; }
    public required float VideoSigma { get; init; }
    public required float AudioSigma { get; init; }
    public required Ltx25FrequencyPrecision FrequencyPrecision { get; init; }
    public float[] VideoTimesteps { get; init; }
    public float[] AudioTimesteps { get; init; }
    public float[] KeyframeMask { get; init; }
    /// <summary>Sampling metadata: one value per video token, 0 freezes a clean
    /// image and 1 fully denoises it. The transformer receives sigma times this mask.</summary>
    public float[] VideoDenoiseMask { get; init; }
    public float[] CleanVideo { get; init; }
    /// <summary>Enable the separately loaded Ingredients IC-LoRA only for
    /// identity reference generation. Ordinary image/keyframe modes leave it off.</summary>
    public bool UseReferenceLora { get; init; }
    public CancellationToken CancellationToken { get; init; }
}

public sealed record Ltx25DenoiserOutput(float[] VideoVelocity, float[] AudioVelocity, int ExecutedBlocks);

/// <summary>Experimental TensorSharp-owned LTX 2.5 AV transformer component.
/// The graph follows the original Lightricks source, including whole-inner Q/K
/// RMS normalization, head gates, per-head split RoPE, nine-way AdaLN and frozen
/// pre-cross-attention stream states. It is not registered as a generation
/// pipeline. Text conditioning, masks, VAEs and generation are separate gates.</summary>
public sealed class Ltx25Denoiser : IDisposable
{
    private readonly GgufFile _file;
    private readonly DirectContext _ctx;
    private readonly CudaLtx25ProjectionSession _cudaProjection;
    private readonly CudaLtx25AttentionSession _cudaAttention;
    private readonly bool _forceCudaAttention;
    private readonly bool _cudaFusedHeadGates;
    private Ltx25CudaFilePrefault _cudaPrefault;
    private readonly Dictionary<string, DirectLinear> _linears = new(StringComparer.Ordinal);
    private readonly Dictionary<string, Tensor> _weights = new(StringComparer.Ordinal);
    private readonly object _sync = new();
    private readonly Ltx25LoraAdapter _referenceLora;
    private bool _useReferenceLora;
    private CancellationToken _cancellationToken;
    private RopeCache _ropeCache;
    private bool _disposed;
    public Ltx25Config Config { get; }
    public long OptimizedCudaProjections => _cudaProjection?.ExecutedProjections ?? 0;
    public string OptimizedCudaModulePath => _cudaProjection?.ModulePath;
    public bool OptimizedCudaModuleAvailable => _cudaProjection?.IsAvailable == true;
    public long OptimizedCudaAttentions => _cudaAttention?.ExecutedAttentions ?? 0;
    public long OptimizedCudaHeadGates => _cudaAttention?.ExecutedHeadGates ?? 0;
    public long CudaAttentionWorkspaceBytes => _cudaAttention?.ScoreCapacityBytes ?? 0;
    public long OptimizedCudaF16Projections => _cudaProjection?.ExecutedF16Projections ?? 0;
    public long OptimizedCudaF16Dequantizations => _cudaProjection?.ExecutedF16Dequantizations ?? 0;
    public bool OptimizedCudaF16Available => _cudaProjection?.F16DequantAvailable == true;
    public long CudaProjectionWorkspaceBytes => _cudaProjection?.F16ScratchBytes ?? 0;
    public long CudaPrefaultReadBytes => _cudaPrefault?.ReadBytes ?? 0;
    public long CudaPrefaultResidentBytes => _cudaPrefault?.ResidentBytes ?? 0;
    public long CudaPrefaultRanges => _cudaPrefault?.Ranges ?? 0;
    public double CudaPrefaultMilliseconds => _cudaPrefault?.Milliseconds ?? 0;
    public string CudaPrefaultFailure => _cudaPrefault?.Failure;
    public long CudaSourceUploadedBytes => _cudaPrefault?.SourceUploadedBytes ?? 0;
    public long CudaSourceUploadedWeights => _cudaPrefault?.SourceUploadedWeights ?? 0;
    public long CudaSourcePinnedBytes => _cudaPrefault?.SourcePinnedBytes ?? 0;
    public double CudaSourceUploadMilliseconds => _cudaPrefault?.SourceUploadMilliseconds ?? 0;
    public bool ReferenceLoraLoaded => _referenceLora != null;
    public int ReferenceLoraProjections => _referenceLora?.ProjectionCount ?? 0;
    internal int CachedReferenceLoraProjections => _referenceLora?.CachedProjectionCount ?? 0;

    // Internal diagnostic observer, installed before execution by validation
    // tools. Arrays are borrowed and must not be mutated. Null on normal models;
    // called after finite output readback and before the sampler advances state.
    internal Action<Ltx25DenoiserInput, Ltx25DenoiserOutput> ForwardObserver { get; set; }

    // Synchronizing between phases changes execution overlap. This diagnostic
    // ranks phases only; its totals must never be reported as throughput.
    private sealed class StageTiming : IDisposable
    {
        private readonly DirectContext _context;
        private readonly Dictionary<string, double> _milliseconds = new(StringComparer.Ordinal);
        private long _previous;
        private StageTiming(DirectContext context)
        {
            _context = context;
            context.CudaAllocator.Synchronize();
            _previous = Stopwatch.GetTimestamp();
        }
        public static StageTiming Start(DirectContext context) => context.IsCuda &&
            Environment.GetEnvironmentVariable("TS_LTX25_STAGE_TIMING") == "1" ? new(context) : null;
        public void Mark(string stage)
        {
            _context.CudaAllocator.Synchronize();
            long now = Stopwatch.GetTimestamp();
            _milliseconds[stage] = _milliseconds.GetValueOrDefault(stage) + (now - _previous) * 1000.0 / Stopwatch.Frequency;
            _previous = now;
        }
        public void Dispose() => Console.WriteLine("[ltx25-profile] synchronized CUDA host phases (diagnostic only): " +
            string.Join(" ", _milliseconds.Select(pair => pair.Key + "=" + pair.Value.ToString("F3", System.Globalization.CultureInfo.InvariantCulture) + "ms")));
    }

    public Ltx25Denoiser(string checkpoint, IAllocator allocator, string referenceLoraPath = null)
    {
        ArgumentNullException.ThrowIfNull(allocator);
        _file = new GgufFile(checkpoint);
        try
        {
            Config = Ltx25Config.FromGguf(_file);
            _ctx = new DirectContext(allocator);
            if (referenceLoraPath != null) _referenceLora = new(referenceLoraPath, _file, Config, allocator);
            if (allocator is CudaAllocator cuda)
            {
                _cudaProjection = new(cuda, Environment.GetEnvironmentVariable("TS_LTX25_CUDA_REQUIRE_TILE") == "1");
                string attentionPolicy = Environment.GetEnvironmentVariable("TS_LTX25_CUDA_ATTN");
                _forceCudaAttention = attentionPolicy == "1";
                _cudaFusedHeadGates = Environment.GetEnvironmentVariable("TS_LTX25_CUDA_FUSED_HEAD_GATE") != "0";
                if (attentionPolicy != "0")
                    _cudaAttention = new(cuda);
            }
        }
        catch { _cudaAttention?.Dispose(); _cudaProjection?.Dispose(); _referenceLora?.Dispose(); _ctx?.Dispose(); _file.Dispose(); throw; }
    }

    private DirectLinear Linear(string name)
    {
        if (_linears.TryGetValue(name, out var value)) return value;
        string p = Config.Prefix + name;
        value = DirectLinear.FromGguf(_ctx, _file, p + ".weight",
            _file.Tensors.ContainsKey(p + ".bias") ? p + ".bias" : null, ltxProjection: _cudaProjection);
        try
        {
            if (_ctx.IsCuda && !_file.IsSplit && Environment.GetEnvironmentVariable("TS_LTX25_CUDA_PREFAULT") != "0")
            {
                var info = _file.Tensors[p + ".weight"];
                _cudaPrefault ??= new(_file);
                _cudaPrefault.Prepare(_ctx.CudaAllocator, value.CacheKey, checked(_file.DataOffset + (long)info.Offset),
                    _file.GetTensorByteCount(info), (int)info.Type, value.InDim, value.OutDim, _cancellationToken);
            }
        }
        catch { value.Dispose(); throw; }
        _linears.Add(name, value);
        return value;
    }

    private Tensor Project(string name, Tensor input)
    {
        _cancellationToken.ThrowIfCancellationRequested();
        var output = Linear(name).Forward(input);
        try
        {
            if (_useReferenceLora) _referenceLora.Add(name, input, output, _cancellationToken);
            return output;
        }
        catch { output.Dispose(); throw; }
    }

    private Tensor Weight(string name)
    {
        if (_weights.TryGetValue(name, out var value)) return value;
        string full = Config.Prefix + name;
        value = _ctx.Own(_ctx.FromFloats(DirectOps.DequantTensor(_file, full),
            _file.Tensors[full].Shape.Reverse().Select(dimension => checked((long)dimension)).ToArray()));
        _weights.Add(name, value);
        return value;
    }

    private sealed class TimeState : IDisposable
    {
        public Tensor Main, Embedded, Prompt, CrossScaleShift, CrossGate;
        public void Dispose()
        {
            Main?.Dispose(); Embedded?.Dispose(); Prompt?.Dispose(); CrossScaleShift?.Dispose(); CrossGate?.Dispose();
        }
    }

    private (Tensor Coefficients, Tensor Embedded) Ada(string name, float[] timesteps, float multiplier)
    {
        var features = TimestepFeatures(timesteps, multiplier);
        using var input = _ctx.FromFloats(features, timesteps.Length, Ltx25Config.TimestepDimension);
        using var first = Project(name + ".emb.timestep_embedder.linear_1", input);
        using var act = DirectOps.Silu(_ctx, first);
        var embedded = Project(name + ".emb.timestep_embedder.linear_2", act);
        try
        {
            using var hidden = DirectOps.Silu(_ctx, embedded);
            return (Project(name + ".linear", hidden), embedded);
        }
        catch { embedded.Dispose(); throw; }
    }

    /// <summary>Original cosine-first 256-feature timestep encoding, before
    /// the learned SiLU MLP. Inputs are sigma, scaled once by the given factor.</summary>
    public static float[] TimestepFeatures(ReadOnlySpan<float> timesteps, float multiplier = Ltx25Config.TimestepMultiplier)
    {
        if (timesteps.Length == 0 || !float.IsFinite(multiplier)) throw new ArgumentException("Invalid LTX timesteps.");
        const int half = Ltx25Config.TimestepDimension / 2;
        var output = new float[checked(timesteps.Length * Ltx25Config.TimestepDimension)];
        for (int row = 0; row < timesteps.Length; ++row)
        {
            float time = timesteps[row] * multiplier;
            if (!float.IsFinite(time)) throw new ArgumentException("LTX timestep must be finite.");
            for (int index = 0; index < half; ++index)
            {
                float phase = time * MathF.Exp(-MathF.Log(10000f) * index / half);
                (output[row * 256 + half + index], output[row * 256 + index]) = MathF.SinCos(phase);
            }
        }
        return output;
    }

    private TimeState PrepareTime(bool video, float sigma, float crossSigma, float[] timesteps)
    {
        var state = new TimeState();
        try
        {
            string p = video ? "" : "audio_";
            (state.Main, state.Embedded) = Ada(p + "adaln_single", timesteps, Config.TimestepScale);
            (state.Prompt, var promptEmbedded) = Ada(p + "prompt_adaln_single", [sigma], Config.TimestepScale);
            promptEmbedded.Dispose();
            (state.CrossScaleShift, var crossEmbedded) = Ada("av_ca_" + (video ? "video" : "audio") + "_scale_shift_adaln_single",
                timesteps, Config.TimestepScale);
            crossEmbedded.Dispose();
            (state.CrossGate, var gateEmbedded) = Ada("av_ca_" + (video ? "a2v" : "v2a") + "_gate_adaln_single",
                [crossSigma], Config.CrossGateTimestepScale);
            gateEmbedded.Dispose();
            return state;
        }
        catch { state.Dispose(); throw; }
    }

    private static Tensor ModulationRow(Tensor dynamic, Tensor table, int index, int coefficients, int dim)
    {
        using var matrix = dynamic.View(dynamic.Sizes[0], coefficients, dim);
        using var column = matrix.Narrow(1, index, 1);
        using var contiguous = Ops.NewContiguous(column);
        using var row = contiguous.View(dynamic.Sizes[0], dim);
        using var staticRow = table.Narrow(0, index, 1);
        return Ops.Add(null, row, staticRow);
    }

    private static Tensor Modulate(Tensor input, Tensor shift, Tensor scale)
    {
        using var scalePlusOne = Ops.Add(null, scale, 1f);
        using var scaled = Ops.Mul(null, input, scalePlusOne);
        return Ops.Add(null, scaled, shift);
    }

    private static void Residual(Tensor target, Tensor update, Tensor gate)
    {
        using var gated = Ops.Mul(null, update, gate);
        Ops.Add(target, target, gated);
    }

    private Tensor Norm(Tensor input) => DirectOps.RmsNorm(_ctx, input, _ctx.Ones(input.Sizes[1]), Ltx25Config.NormEpsilon);

    private sealed class DeviceRope : IDisposable
    {
        public readonly Tensor Cos, Sin;
        public readonly int Tokens, Heads, HeadDim;
        public DeviceRope(DirectContext ctx, Ltx25Rope rope)
        {
            Tokens = rope.Tokens; Heads = rope.Heads; HeadDim = rope.HeadDim;
            Cos = ctx.FromFloats(rope.Cosine, Tokens, Heads, 1, HeadDim / 2);
            try { Sin = ctx.FromFloats(rope.Sine, Tokens, Heads, 1, HeadDim / 2); }
            catch { Cos.Dispose(); throw; }
        }
        public void Dispose() { Cos.Dispose(); Sin.Dispose(); }
    }

    // Position tables do not depend on sigma. Retain one generation's tables
    // across Euler steps instead of repeating millions of trig evaluations and
    // host-to-device transfers. Snapshot the caller-owned coordinates so edits
    // to a reused input grid cannot silently reuse stale tables.
    private sealed class RopeCache : IDisposable
    {
        private readonly float[] _videoBoundaries, _audioBoundaries;
        private readonly Ltx25FrequencyPrecision _precision;
        public readonly DeviceRope Video, Audio, CrossVideo;
        public RopeCache(DirectContext ctx, Ltx25Config config, Ltx25DenoiserInput input)
        {
            _videoBoundaries = (float[])input.VideoPositions.Boundaries.Clone();
            _audioBoundaries = (float[])input.AudioPositions.Boundaries.Clone();
            _precision = input.FrequencyPrecision;
            try
            {
                Video = new(ctx, Ltx25Geometry.BuildRope(input.VideoPositions, config.VideoDim, config.VideoHeads, _precision));
                Audio = new(ctx, Ltx25Geometry.BuildRope(input.AudioPositions, config.AudioDim, config.AudioHeads, _precision));
                CrossVideo = new(ctx, Ltx25Geometry.BuildRope(Ltx25Geometry.TemporalPositions(input.VideoPositions),
                    config.AudioDim, config.AudioHeads, _precision));
            }
            catch { Dispose(); throw; }
        }
        public bool Matches(Ltx25DenoiserInput input) => _precision == input.FrequencyPrecision &&
            _videoBoundaries.AsSpan().SequenceEqual(input.VideoPositions.Boundaries) &&
            _audioBoundaries.AsSpan().SequenceEqual(input.AudioPositions.Boundaries);
        public void Dispose() { Video?.Dispose(); Audio?.Dispose(); CrossVideo?.Dispose(); }
    }

    private RopeCache PrepareRope(Ltx25DenoiserInput input)
    {
        if (_ropeCache?.Matches(input) == true) return _ropeCache;
        // Release the old generation before constructing potentially large
        // replacement tables, keeping peak residency bounded to one geometry.
        _ropeCache?.Dispose();
        _ropeCache = null;
        return _ropeCache = new RopeCache(_ctx, Config, input);
    }

    private void ApplyRope(Tensor input, DeviceRope rope)
    {
        if (input.Sizes[0] != rope.Tokens || input.Sizes[1] != rope.Heads * rope.HeadDim)
            throw new ArgumentException("LTX per-head rotary dimensions differ.");
        if (_ctx.IsAccelerated && Environment.GetEnvironmentVariable("TS_LTX25_FUSED_ROPE") != "0")
        {
            // Each LTX head has its own frequencies. Treat token/head pairs as
            // independent rows for the existing split-half fused rotary kernel.
            int rows = checked(rope.Tokens * rope.Heads), half = rope.HeadDim / 2;
            using var flat = input.View(rows, rope.HeadDim);
            using var cos = rope.Cos.View(checked(rows * half));
            using var sin = rope.Sin.View(checked(rows * half));
            if (_ctx.IsCuda && CudaFusedOps.TryNeoXRoPEFlatInPlace(flat, cos, sin, 1, rows, rope.HeadDim, half) ||
                _ctx.IsMlx && MlxFusedOps.TryNeoXRoPEFlatInPlace(flat, cos, sin, 1, rows, rope.HeadDim, half)) return;
        }
        using var view = input.View(rope.Tokens, rope.Heads, 2, rope.HeadDim / 2);
        using var first = view.Narrow(2, 0, 1);
        using var second = view.Narrow(2, 1, 1);
        using var oldFirst = Ops.NewContiguous(first);
        using var oldSecond = Ops.NewContiguous(second);
        using var firstCos = Ops.Mul(null, oldFirst, rope.Cos);
        using var secondSin = Ops.Mul(null, oldSecond, rope.Sin);
        using var firstResult = Ops.Sub(null, firstCos, secondSin);
        using var secondCos = Ops.Mul(null, oldSecond, rope.Cos);
        using var firstSin = Ops.Mul(null, oldFirst, rope.Sin);
        using var secondResult = Ops.Add(null, secondCos, firstSin);
        Ops.Copy(first, firstResult);
        Ops.Copy(second, secondResult);
    }

    private Tensor Attention(string name, Tensor query, Tensor context, int heads, int headDim,
        DeviceRope queryRope = null, DeviceRope keyRope = null)
    {
        using var qRaw = Project(name + ".to_q", query);
        using var kRaw = Project(name + ".to_k", context);
        using var value = Project(name + ".to_v", context);
        // LTX RMS normalizes all heads together before splitting them.
        using var q = DirectOps.RmsNorm(_ctx, qRaw, Weight(name + ".q_norm.weight"), Ltx25Config.NormEpsilon);
        using var k = DirectOps.RmsNorm(_ctx, kRaw, Weight(name + ".k_norm.weight"), Ltx25Config.NormEpsilon);
        if (queryRope != null) ApplyRope(q, queryRope);
        if (keyRope != null) ApplyRope(k, keyRope);
        using var attention = Attend(name, q, k, value, heads, headDim);
        using var logits = Project(name + ".to_gate_logits", query);
        if (_cudaFusedHeadGates && _cudaAttention?.TryApplyHeadGates(attention, logits, heads, headDim) == true)
            return Project(name + ".to_out.0", attention);
        using var gates = Ops.Sigmoid(null, logits);
        Ops.Mul(gates, gates, 2f);
        using var headGates = gates.View(query.Sizes[0], heads, 1);
        using var headOutput = attention.View(query.Sizes[0], heads, headDim);
        // CPU's generic iterator requires explicitly broadcast strides; passing
        // the smaller tensor directly may truncate the multiply or fail SIMD
        // alignment when token*head count is not a vector-width multiple.
        using var expandedGates = headGates.Expand(headOutput.Sizes);
        if (_ctx.IsCuda)
        {
            // CUDA binary kernels accept contiguous operands or row vectors.
            // Materialize this column broadcast with the device strided-copy
            // kernel so every attention avoids a CPU fallback/readback.
            using var contiguousGates = Ops.NewContiguous(expandedGates);
            Ops.Mul(headOutput, headOutput, contiguousGates);
        }
        else Ops.Mul(headOutput, headOutput, expandedGates);
        return Project(name + ".to_out.0", attention);
    }

    private Tensor Attend(string name, Tensor query, Tensor key, Tensor value, int heads, int headDim)
    {
        float scale = 1f / MathF.Sqrt(headDim);
        // At representative video sizes, batched pedantic F32 products avoid
        // the scalar streaming kernel's compute bottleneck. Short attention
        // keeps its launch-lean route; env1 also exercises small numerical tests.
        if (_cudaAttention != null && (_forceCudaAttention || query.Sizes[0] >= 512 && key.Sizes[0] >= 512) &&
            (name.EndsWith(".attn1", StringComparison.Ordinal) ||
            name.EndsWith(".attn2", StringComparison.Ordinal)))
        {
            var output = _ctx.NewF32(query.Sizes);
            try
            {
                if (_cudaAttention.TryAttention(output, query, key, value, heads, headDim, scale)) return output;
            }
            catch { output.Dispose(); throw; }
            output.Dispose();
        }
        return DirectOps.Attention(_ctx, query, key, value, heads, headDim, scale);
    }

    private void SelfAndText(string block, bool video, Tensor state, Tensor context, TimeState time, DeviceRope rope)
    {
        string p = video ? "" : "audio_";
        int dim = video ? Config.VideoDim : Config.AudioDim;
        int heads = video ? Config.VideoHeads : Config.AudioHeads;
        int headDim = dim / heads;
        Tensor table = Weight(block + p + "scale_shift_table");
        using (var shift = ModulationRow(time.Main, table, 0, 9, dim))
        using (var scale = ModulationRow(time.Main, table, 1, 9, dim))
        using (var gate = ModulationRow(time.Main, table, 2, 9, dim))
        using (var norm = Norm(state))
        using (var input = Modulate(norm, shift, scale))
        using (var update = Attention(block + p + "attn1", input, input, heads, headDim, rope, rope))
            Residual(state, update, gate);
        using (var shift = ModulationRow(time.Main, table, 6, 9, dim))
        using (var scale = ModulationRow(time.Main, table, 7, 9, dim))
        using (var gate = ModulationRow(time.Main, table, 8, 9, dim))
        using (var norm = Norm(state))
        using (var query = Modulate(norm, shift, scale))
        using (var shiftContext = ModulationRow(time.Prompt, Weight(block + p + "prompt_scale_shift_table"), 0, 2, dim))
        using (var scaleContext = ModulationRow(time.Prompt, Weight(block + p + "prompt_scale_shift_table"), 1, 2, dim))
        using (var projectedContext = Modulate(context, shiftContext, scaleContext))
        using (var update = Attention(block + p + "attn2", query, projectedContext, heads, headDim))
            Residual(state, update, gate);
    }

    private Tensor CrossInput(string block, bool video, Tensor norm, TimeState time, int from)
    {
        int dim = video ? Config.VideoDim : Config.AudioDim;
        using var table = Weight(block + "scale_shift_table_a2v_ca_" + (video ? "video" : "audio")).Narrow(0, 0, 4);
        using var scale = ModulationRow(time.CrossScaleShift, table, from, 4, dim);
        using var shift = ModulationRow(time.CrossScaleShift, table, from + 1, 4, dim);
        return Modulate(norm, shift, scale);
    }

    private Tensor CrossGate(string block, bool video, TimeState time)
    {
        using var table = Weight(block + "scale_shift_table_a2v_ca_" + (video ? "video" : "audio")).Narrow(0, 4, 1);
        return ModulationRow(time.CrossGate, table, 0, 1, video ? Config.VideoDim : Config.AudioDim);
    }

    private void CrossStreams(string block, Tensor video, Tensor audio, TimeState vt, TimeState at,
        DeviceRope videoRope, DeviceRope audioRope)
    {
        // Freeze BOTH normalized streams before mutating either residual.
        // V2A must never see the already-updated A2V video state.
        using var videoNorm = Norm(video);
        using var audioNorm = Norm(audio);
        using (var query = CrossInput(block, true, videoNorm, vt, 0))
        using (var context = CrossInput(block, false, audioNorm, at, 0))
        using (var gate = CrossGate(block, true, vt))
        using (var update = Attention(block + "audio_to_video_attn", query, context,
            Config.AudioHeads, Config.AudioHeadDim, videoRope, audioRope))
            Residual(video, update, gate);
        using (var query = CrossInput(block, false, audioNorm, at, 2))
        using (var context = CrossInput(block, true, videoNorm, vt, 2))
        using (var gate = CrossGate(block, false, at))
        using (var update = Attention(block + "video_to_audio_attn", query, context,
            Config.AudioHeads, Config.AudioHeadDim, audioRope, videoRope))
            Residual(audio, update, gate);
    }

    private void FeedForward(string block, bool video, Tensor state, TimeState time)
    {
        string p = video ? "" : "audio_";
        int dim = video ? Config.VideoDim : Config.AudioDim;
        Tensor table = Weight(block + p + "scale_shift_table");
        using var shift = ModulationRow(time.Main, table, 3, 9, dim);
        using var scale = ModulationRow(time.Main, table, 4, 9, dim);
        using var gate = ModulationRow(time.Main, table, 5, 9, dim);
        using var norm = Norm(state);
        using var input = Modulate(norm, shift, scale);
        using var wide = Project(block + p + "ff.net.0.proj", input);
        using var activation = DirectOps.Gelu(_ctx, wide);
        using var update = Project(block + p + "ff.net.2", activation);
        Residual(state, update, gate);
    }

    private Tensor Output(bool video, Tensor state, Tensor embedded)
    {
        string p = video ? "" : "audio_";
        Tensor table = Weight(p + "scale_shift_table");
        using var shiftRow = table.Narrow(0, 0, 1);
        using var scaleRow = table.Narrow(0, 1, 1);
        using var shift = Ops.Add(null, embedded, shiftRow);
        using var scale = Ops.Add(null, embedded, scaleRow);
        using var norm = DirectOps.LayerNorm(_ctx, state, null, null, Ltx25Config.NormEpsilon);
        using var input = Modulate(norm, shift, scale);
        return Project(p + "proj_out", input);
    }

    /// <param name="blockLimit">A diagnostic partial-checkpoint limit. Output
    /// records the actual count. Omit it to execute the complete denoiser.</param>
    public Ltx25DenoiserOutput Forward(Ltx25DenoiserInput input, int? blockLimit = null)
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            Validate(input);
            input.CancellationToken.ThrowIfCancellationRequested();
            if (input.UseReferenceLora && _referenceLora == null)
                throw new InvalidOperationException("Ingredients reference generation requires the LTX 2.5 Ingredients IC-LoRA adapter.");
            _useReferenceLora = input.UseReferenceLora;
            _cancellationToken = input.CancellationToken;
            int blocks = blockLimit ?? Config.Layers;
            if (blocks <= 0 || blocks > Config.Layers) throw new ArgumentOutOfRangeException(nameof(blockLimit));
            using var timing = StageTiming.Start(_ctx);
            using var videoInput = _ctx.FromFloats(input.Video, input.VideoTokens, Config.VideoChannels);
            using var audioInput = _ctx.FromFloats(input.Audio, input.AudioTokens, Config.AudioChannels);
            using var video = Project("patchify_proj", videoInput);
            using var audio = Project("audio_patchify_proj", audioInput);
            if (input.KeyframeMask != null)
            {
                if (!Config.KeyframeEmbedding) throw new ArgumentException("Checkpoint has no keyframe embedding.");
                using var mask = _ctx.FromFloats(input.KeyframeMask.Select(value => value > 0 ? 1f : 0f).ToArray(), input.VideoTokens, 1);
                using var expandedMask = mask.Expand(input.VideoTokens, Config.VideoDim);
                using var embedding = Weight("keyframes_abs_pos_embedding").Expand(input.VideoTokens, Config.VideoDim);
                if (_ctx.IsCuda)
                {
                    using var contiguousMask = Ops.NewContiguous(expandedMask);
                    using var contiguousEmbedding = Ops.NewContiguous(embedding);
                    using var absolute = Ops.Mul(null, contiguousMask, contiguousEmbedding);
                    Ops.Add(video, video, absolute);
                }
                else
                {
                    using var absolute = Ops.Mul(null, expandedMask, embedding);
                    Ops.Add(video, video, absolute);
                }
            }
            using var videoContext = _ctx.FromFloats(input.VideoContext, input.VideoContextTokens, Config.VideoContextDim);
            using var audioContext = _ctx.FromFloats(input.AudioContext, input.AudioContextTokens, Config.AudioContextDim);
            using var vt = PrepareTime(true, input.VideoSigma, input.AudioSigma, input.VideoTimesteps ?? [input.VideoSigma]);
            using var at = PrepareTime(false, input.AudioSigma, input.VideoSigma, input.AudioTimesteps ?? [input.AudioSigma]);
            var rope = PrepareRope(input);
            timing?.Mark("prelude");
            for (int i = 0; i < blocks; ++i)
            {
                string block = $"transformer_blocks.{i}.";
                SelfAndText(block, true, video, videoContext, vt, rope.Video);
                timing?.Mark("video_self_text");
                SelfAndText(block, false, audio, audioContext, at, rope.Audio);
                timing?.Mark("audio_self_text");
                CrossStreams(block, video, audio, vt, at, rope.CrossVideo, rope.Audio);
                timing?.Mark("cross_streams");
                FeedForward(block, true, video, vt);
                timing?.Mark("video_ffn");
                FeedForward(block, false, audio, at);
                timing?.Mark("audio_ffn");
            }
            using var videoOutput = Output(true, video, vt.Embedded);
            using var audioOutput = Output(false, audio, at.Embedded);
            var videoValues = DirectOps.ToArray(videoOutput);
            var audioValues = DirectOps.ToArray(audioOutput);
            timing?.Mark("output_readback");
            if (videoValues.Any(value => !float.IsFinite(value)) || audioValues.Any(value => !float.IsFinite(value)))
                throw new InvalidOperationException("LTX denoiser produced nonfinite velocity.");
            var result = new Ltx25DenoiserOutput(videoValues, audioValues, blocks);
            ForwardObserver?.Invoke(input, result);
            return result;
        }
    }

    private void Validate(Ltx25DenoiserInput input)
    {
        ArgumentNullException.ThrowIfNull(input);
        static void Rows(float[] values, int rows, int width)
        {
            if (rows <= 0 || values == null || values.Length != checked(rows * width) ||
                values.Any(value => !float.IsFinite(value))) throw new ArgumentException("Invalid LTX token rows.");
        }
        static void Times(float sigma, float[] values, int tokens)
        {
            if (!float.IsFinite(sigma) || sigma < 0 || sigma > 1 || values != null &&
                ((values.Length != 1 && values.Length != tokens) || values.Any(value => !float.IsFinite(value) || value < 0 || value > 1)))
                throw new ArgumentException("LTX sigma/timesteps must be finite [0,1] scalar or per-token values.");
        }
        Rows(input.Video, input.VideoTokens, Config.VideoChannels); Rows(input.Audio, input.AudioTokens, Config.AudioChannels);
        Rows(input.VideoContext, input.VideoContextTokens, Config.VideoContextDim); Rows(input.AudioContext, input.AudioContextTokens, Config.AudioContextDim);
        Times(input.VideoSigma, input.VideoTimesteps, input.VideoTokens); Times(input.AudioSigma, input.AudioTimesteps, input.AudioTokens);
        if (input.VideoPositions == null || input.VideoPositions.Tokens != input.VideoTokens || input.VideoPositions.Axes != 3 ||
            input.AudioPositions == null || input.AudioPositions.Tokens != input.AudioTokens || input.AudioPositions.Axes != 1 ||
            !Enum.IsDefined(input.FrequencyPrecision)) throw new ArgumentException("LTX latent position grids differ.");
        if (input.KeyframeMask != null && (input.KeyframeMask.Length != input.VideoTokens || input.KeyframeMask.Any(value => !float.IsFinite(value))))
            throw new ArgumentException("Invalid LTX keyframe mask.");
    }

    /// <summary>Evict only unshared direct linear cache entries. Host GGUF
    /// descriptors and small model tensors remain valid for the next forward.</summary>
    public void ReleaseDeviceResidency()
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            _ropeCache?.Dispose(); _ropeCache = null;
            _cudaAttention?.ReleaseWorkspace();
            _cudaProjection?.ReleaseWorkspace();
            _ctx.ReleaseDeviceResidency();
            _referenceLora?.ReleaseDeviceResidency();
        }
    }

    public void Dispose()
    {
        lock (_sync)
        {
            if (_disposed) return;
            _disposed = true;
            _ropeCache?.Dispose(); _ropeCache = null;
            _cudaAttention?.Dispose();
            _cudaProjection?.Dispose();
            _cudaPrefault?.Dispose();
            _referenceLora?.Dispose();
            _ctx.Dispose();
            _file.Dispose();
            _linears.Clear(); _weights.Clear();
        }
    }
}
