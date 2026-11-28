// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using TensorSharp.GGML;

namespace TensorSharp.Models;

/// <summary>
/// Muse Spark GGUF decoder. GGUF conversion has already folded delta norms, MetaP
/// Q/router scales and residual gates into the weights. Only sliding layers use
/// interleaved RoPE. Each expert is normalized BEFORE its routing weight is applied.
/// </summary>
public partial class MuseSparkModel : ModelBase
{
    private readonly int _headDim, _ropeDim, _latentSize, _expertFfnLength;
    private readonly int _numExperts, _numExpertsUsed, _slidingWindow, _prefillChunk;
    private readonly float _postNormEps, _logitScale, _softcap, _expertScale;
    private readonly bool _expertWeightsNorm;
    private readonly bool[] _isSwa;
    private readonly Dictionary<int, Tensor> _ones = new();
    private Tensor[] _kvCacheK, _kvCacheV;
    private int _cacheCapacity;
    private readonly string[] _prefixes;

    public MuseSparkModel(string ggufPath, BackendType backend, int tpDegree = 1,
        ITensorParallelGroup tpGroup = null, int layerSplitDegree = 1)
        : this(ggufPath, backend, tpDegree, tpGroup, layerSplitDegree, null)
    {
    }

    // A supplied snapshot allows deterministic placement tests without reserving
    // device memory. The public constructor always queries the selected devices.
    internal MuseSparkModel(string ggufPath, BackendType backend, int tpDegree,
        ITensorParallelGroup tpGroup, int layerSplitDegree, long[] tensorParallelFreeBytes)
        : base(ggufPath, ValidateBackend(backend, tpDegree, tpGroup, layerSplitDegree),
            // The native executor owns every GPU it uses; the managed base keeps only
            // device 0's context (vision tower, tokenizer-side helpers).
            NativeRequested(backend, tpDegree, tpGroup) ? 1 : tpDegree,
            NativeRequested(backend, tpDegree, tpGroup) ? null : tpGroup,
            NativeRequested(backend, tpDegree, tpGroup) ? 1 : layerSplitDegree)
    {
        _sparkTpPlanningFreeBytes = tensorParallelFreeBytes == null ? null : (long[])tensorParallelFreeBytes.Clone();
        try
        {
        string arch = _gguf.GetString("general.architecture") ?? "muse-spark";
        Config = new ModelConfig { Architecture = "muse-spark" };
        MetadataArchitecture = arch;
        ParseBaseConfig();
        _headDim = Config.HeadDim;
        _ropeDim = (int)_gguf.GetUint32($"{arch}.rope.dimension_count", (uint)_headDim);
        _latentSize = (int)_gguf.GetUint32($"{arch}.moe_latent_size");
        _expertFfnLength = (int)_gguf.GetUint32($"{arch}.expert_feed_forward_length");
        _numExperts = Config.NumExperts = (int)_gguf.GetUint32($"{arch}.expert_count");
        _numExpertsUsed = Config.NumExpertsUsed = (int)_gguf.GetUint32($"{arch}.expert_used_count");
        _slidingWindow = Config.SlidingWindow = (int)_gguf.GetUint32($"{arch}.attention.sliding_window");
        _postNormEps = _gguf.GetFloat32($"{arch}.post_norm_epsilon");
        _logitScale = _gguf.GetFloat32($"{arch}.logit_scale", 1f);
        _softcap = _gguf.GetFloat32($"{arch}.final_logit_softcapping");
        _expertScale = _gguf.GetFloat32($"{arch}.expert_weights_scale", 1f);
        _expertWeightsNorm = _gguf.GetBool($"{arch}.expert_weights_norm", true);
        Config.RopeBase = _gguf.GetFloat32($"{arch}.rope.freq_base_swa", Config.RopeBase);
        _isSwa = _gguf.GetBoolArray($"{arch}.attention.sliding_window_pattern");
        if (_isSwa == null || _isSwa.Length != Config.NumLayers)
            throw new InvalidDataException("Muse Spark requires one sliding-window flag per layer.");
        if (_gguf.GetUint32($"{arch}.expert_gating_func", 2) != 2)
            throw new NotSupportedException("Muse Spark requires sigmoid expert routing.");
        if (Config.NumLayers <= 0 || Config.HiddenSize <= 0 || Config.NumHeads <= 0 ||
            Config.NumKVHeads <= 0 || Config.NumHeads % Config.NumKVHeads != 0 || _headDim <= 0 ||
            (Config.ValueLength > 0 && Config.ValueLength != _headDim) ||
            _latentSize <= 0 || _expertFfnLength <= 0 || _numExpertsUsed <= 0 || _numExpertsUsed > _numExperts ||
            _slidingWindow <= 0 || _ropeDim <= 0 || _ropeDim > _headDim || _ropeDim % 2 != 0 ||
            !(Config.Eps > 0) || !(_postNormEps > 0) || !(Config.RopeBase > 0) || !(Config.RopeScale > 0) ||
            !float.IsFinite(_expertScale) || !float.IsFinite(_logitScale) || _softcap < 0)
            throw new InvalidDataException("Invalid Muse Spark attention, norm or expert geometry.");
        _prefillChunk = int.TryParse(Environment.GetEnvironmentVariable("TS_MUSE_SPARK_PREFILL_CHUNK"), out int chunk)
            && chunk > 0 ? chunk : 256;
        _prefixes = new string[Config.NumLayers];
        for (int l = 0; l < _prefixes.Length; l++) _prefixes[l] = $"blk.{l}.";
        ParseTokenizer();
        ValidateWeights();
        if (NativeRequested(backend, tpDegree, tpGroup))
        {
            InitNativeExecutor(ggufPath, backend, layerSplitDegree, tpDegree);
            return;
        }
        PreflightLayerSplitCapacity();
        PreflightSparkTensorParallel();
        LoadWeights();
        // Choose from the complete checkpoint before TP removes matrices from
        // the unsharded dictionaries.
        ApplyModelAlignedKvCacheDefault(_quantWeights);
        InitializeSparkTensorParallel();
        ConfigureSparkTensorParallelHostPool();
        FuseAttentionProjections();
        if (IsTensorParallel) PrepareCudaQuantizedWeightsForInferenceTP();
        else PrepareCudaQuantizedWeightsForInference();
        ReleasePreloadedGpuSourcePages();
        if (IsTensorParallel) ReleaseTensorParallelGpuSourcePages();
        _maxContextLength = ResolveConfiguredContextLength();
        _kvCacheK = new Tensor[Config.NumLayers];
        _kvCacheV = new Tensor[Config.NumLayers];
        EnsureCacheCapacity(ResolveInitialCacheAllocationLength(_maxContextLength, 2048));
        Console.WriteLine($"Muse Spark: {Config.NumLayers} layers, {_numExperts} experts/top-{_numExpertsUsed}, " +
            $"latent {_latentSize}, SWA {_slidingWindow}, {_backend}, layer split {LayerSplitDegree}.");
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    private static BackendType ValidateBackend(BackendType backend, int tp, ITensorParallelGroup group, int split)
    {
        bool tensorParallel = tp > 1 || group?.IsActive == true;
        if (tensorParallel && split > 1)
            throw new NotSupportedException("Muse Spark tensor parallelism and layer split are separate placement modes.");
        if (tensorParallel && backend is not (BackendType.GgmlCuda or BackendType.GgmlVulkan) &&
            !(backend == BackendType.Cpu && group != null))
            throw new NotSupportedException("Muse Spark tensor parallelism requires ggml_cuda or ggml_vulkan (or an explicit CPU test group).");
        if (split > 1 && backend is not (BackendType.GgmlCuda or BackendType.GgmlVulkan))
            throw new NotSupportedException("Muse Spark layer placement requires ggml_cuda or ggml_vulkan.");
        if (MoeCpuOffloadConfig.Current.IsEnabled && backend is not (BackendType.Cpu or BackendType.GgmlCpu or
            BackendType.GgmlCuda or BackendType.GgmlVulkan or BackendType.GgmlMetal))
            throw new NotSupportedException("Muse Spark expert CPU offload requires a GGML backend.");
        return backend;
    }

    // Do not prefault hundreds of gigabytes of sparse offloaded experts: the OS
    // demand-pages selected expert rows from the unchanged GGUF mmap.
    protected override bool ShouldPrefaultWeight(GgufTensorInfo info)
        => !MoeOffload.IsOffloadedExpertWeightName(info.Name);

    protected override bool ShouldPreloadCudaQuantWeightToDevice(string name)
        => base.ShouldPreloadCudaQuantWeightToDevice(name) && !_stackedExpertMemberNames.Contains(name);

    private int LayerRank(int layer) => _layerRanks == null ? 0 : _layerRanks[layer];

    protected override int PreloadRankForWeight(string name)
    {
        if (name.StartsWith("blk.", StringComparison.Ordinal))
        {
            int dot = name.IndexOf('.', 4);
            if (dot > 4 && int.TryParse(name.AsSpan(4, dot - 4), out int layer)) return LayerRank(layer);
        }
        return name.StartsWith("output", StringComparison.Ordinal) ? LayerRank(Config.NumLayers - 1) : 0;
    }

    // Host views also serve expert fallback and cross-rank embedding/LM-head reads.
    protected override bool ShouldRetainCudaHostQuantWeight(string name) => true;

    // Cache growth uses row views; the block-quantized row-copy path is not
    // implemented here. F32/F16 work on every attention and prefix-cache path.
    protected override bool SupportsBlockQuantizedKvCache => false;

    private void ValidateWeights()
    {
        void Require(string name, params ulong[] shape)
        {
            if (!_gguf.Tensors.TryGetValue(name, out var tensor) || !tensor.Shape.AsSpan().SequenceEqual(shape))
                throw new InvalidDataException($"Muse Spark tensor {name} is missing or has incompatible dimensions.");
        }
        ulong h = (ulong)Config.HiddenSize, hd = (ulong)_headDim, q = (ulong)(Config.NumHeads * _headDim);
        ulong kv = (ulong)(Config.NumKVHeads * _headDim), latent = (ulong)_latentSize;
        Require("token_embd.weight", h, (ulong)Config.VocabSize);
        Require("output.weight", h, (ulong)Config.VocabSize);
        Require("output_norm.weight", h);
        foreach (string p in _prefixes)
        {
            foreach (string n in new[] { "attn_norm", "ffn_norm", "post_ffw_norm", "attn_resid_alpha", "attn_resid_beta", "ffn_resid_alpha", "ffn_resid_beta" }) Require(p + n + ".weight", h);
            Require(p + "attn_q.weight", h, q); Require(p + "attn_k.weight", h, kv); Require(p + "attn_v.weight", h, kv);
            Require(p + "attn_gate.weight", h, q); Require(p + "attn_output.weight", q, h);
            Require(p + "attn_q_norm.weight", hd); Require(p + "attn_k_norm.weight", hd);
            Require(p + "ffn_gate_inp.weight", h, (ulong)_numExperts); Require(p + "exp_probs_b.bias", (ulong)_numExperts);
            Require(p + "ffn_latent_down.weight", h, latent); Require(p + "ffn_latent_up.weight", latent, h);
            Require(p + "ffn_pre_exp_norm.weight", latent); Require(p + "ffn_post_exp_norm.weight", latent);
            Require(p + "ffn_gate_exps.weight", latent, (ulong)_expertFfnLength, (ulong)_numExperts);
            Require(p + "ffn_up_exps.weight", latent, (ulong)_expertFfnLength, (ulong)_numExperts);
            Require(p + "ffn_down_exps.weight", (ulong)_expertFfnLength, latent, (ulong)_numExperts);
        }
    }

    private Tensor Ones(int width)
    {
        if (!_ones.TryGetValue(width, out var weight))
        {
            weight = new Tensor(_allocator, DType.Float32, width);
            Ops.Fill(weight, 1f);
            _ones.Add(width, weight);
        }
        return weight;
    }

    private Tensor Norm(Tensor input, float eps, string weight = null)
    {
        var gamma = weight == null ? Ones((int)input.Sizes[^1]) : _weights[weight];
        return UseNormScratch ? ScratchNorm(input, gamma, eps) : Ops.RMSNorm(null, input, gamma, null, eps);
    }

    private Tensor ScaleChannels(Tensor input, string weight)
    {
        using var row = _weights[weight].View(1, input.Sizes[1]);
        using var expanded = row.Expand(input.Sizes);
        return Ops.Mul(null, input, expanded);
    }

    private Tensor Residual(Tensor stream, Tensor branch, string alpha, string beta)
    {
        using var a = ScaleChannels(stream, alpha);
        using var b = ScaleChannels(branch, beta);
        return Ops.Add(null, a, b);
    }

    protected override float[] ForwardCore(int[] tokens)
    {
        if (tokens == null || tokens.Length == 0) throw new ArgumentException("A forward pass needs at least one token.", nameof(tokens));
        if (UsesNativeExecutor) return ForwardNative(tokens);
        if ((long)_cacheSeqLen + tokens.Length > _maxContextLength) throw new InvalidOperationException("Muse Spark context length exceeded.");
        float[] logits = null;
        for (int offset = 0; offset < tokens.Length; offset += _prefillChunk)
        {
            int count = Math.Min(_prefillChunk, tokens.Length - offset);
            int[] part = offset == 0 && count == tokens.Length ? tokens : tokens.AsSpan(offset, count).ToArray();
            logits = ForwardChunk(part, offset + count == tokens.Length);
        }
        return logits;
    }

    private float[] ForwardChunk(int[] tokens, bool outputLogits, bool allLogitsRows = false, float[] hiddenRows = null)
    {
        if (!_sparkPhaseTimingEnabled) return ForwardChunkCore(tokens, outputLogits, allLogitsRows, hiddenRows);
        return TimedForwardChunk(tokens, outputLogits, allLogitsRows, hiddenRows);
    }

    // Keep the capturing delegate in a separate enabled-only method: a C#
    // display class must not be allocated at entry to the disabled fast path.
    private float[] TimedForwardChunk(int[] tokens, bool outputLogits, bool allLogitsRows, float[] hiddenRows)
    {
        return RecordSparkForward("ordinary", tokens.Length, _cacheSeqLen,
            () => ForwardChunkCore(tokens, outputLogits, allLogitsRows, hiddenRows));
    }

    private float[] ForwardChunkCore(int[] tokens, bool outputLogits, bool allLogitsRows, float[] hiddenRows)
    {
        _forwardSw.Start();
        int rows = tokens.Length, start = _cacheSeqLen;
        EnsureCacheCapacity(start + rows);
        bool pinExecution = IsGgmlBackend && (LayerSplitDegree > 1 || IsTensorParallel);
        int priorRank = pinExecution ? GgmlBasicOps.GetActiveRank() : 0;
        bool priorPin = pinExecution && GgmlBasicOps.PinRank(0);
        Tensor hidden = null;
        try
        {
            _sparkPhaseFrame?.Next(MuseSparkPhase.Embedding);
            using (var embeddings = Embedding(tokens))
            {
                DrainPendingVisionEmbeddings(embeddings, start);
                hidden = Norm(embeddings, Config.Eps);
            }
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                _sparkPhaseFrame?.Next(MuseSparkPhase.LayerNorms);
                if (LayerSplitDegree > 1 && GgmlBasicOps.GetActiveRank() != LayerRank(layer))
                {
                    SyncTensorHostCache(hidden);
                    GgmlBasicOps.SetActiveRank(LayerRank(layer));
                }
                string p = _prefixes[layer];
                using var preAttention = Norm(hidden, Config.Eps, p + "attn_norm.weight");
                _sparkPhaseFrame?.Next(MuseSparkPhase.AttentionSingle);
                using var attention = Attention(preAttention, layer, rows, start);
                _sparkPhaseFrame?.Next(MuseSparkPhase.Residual);
                using var ffnInput = PostNormResidual(hidden, attention, p + "attn_resid_alpha.weight", p + "attn_resid_beta.weight");
                hidden.Dispose(); hidden = null;
                _sparkPhaseFrame?.Next(MuseSparkPhase.LayerNorms);
                using var ffnNorm = Norm(ffnInput, Config.Eps, p + "ffn_norm.weight");
                using var moeOutput = MoE(ffnNorm, layer, rows);
                _sparkPhaseFrame?.Next(MuseSparkPhase.Residual);
                hidden = PostNormResidual(ffnInput, moeOutput, p + "ffn_resid_alpha.weight", p + "ffn_resid_beta.weight",
                    p + "post_ffw_norm.weight");
                _sparkPhaseFrame?.Next(MuseSparkPhase.Cleanup);
            }
            _cacheSeqLen += rows;
            if (!outputLogits) return null;
            _sparkPhaseFrame?.Next(MuseSparkPhase.OutputNorm);
            using var last = hidden.Narrow(0, allLogitsRows || hiddenRows != null ? 0 : rows - 1,
                allLogitsRows || hiddenRows != null ? rows : 1);
            using var final = Norm(last, Config.Eps, "output_norm.weight");
            if (hiddenRows != null) TensorToFloatArray(final).CopyTo(hiddenRows, 0);
            using var headInput = final.Narrow(0, allLogitsRows ? 0 : (int)final.Sizes[0] - 1,
                allLogitsRows ? rows : 1);
            _sparkPhaseFrame?.Next(MuseSparkPhase.OutputHead);
            using var logits = LinearForward(headInput, "output.weight");
            _logitsBuffer = TensorToFloatArray(logits);
            for (int i = 0; i < _logitsBuffer.Length; i++)
            {
                float x = _logitsBuffer[i] * _logitScale;
                _logitsBuffer[i] = _softcap > 0 ? _softcap * MathF.Tanh(x / _softcap) : x;
            }
            _sparkPhaseFrame?.Next(MuseSparkPhase.Cleanup);
            return _logitsBuffer;
        }
        finally
        {
            _sparkPhaseFrame?.Next(MuseSparkPhase.Cleanup);
            hidden?.Dispose();
            if (pinExecution) GgmlBasicOps.RestoreRankPin(priorPin, priorRank);
            _forwardCount++;
            _forwardSw.Stop();
        }
    }

    private Tensor HeadNorm(Tensor input, int rows, int heads, string weight = null)
    {
        using var headsView = input.View(rows * heads, _headDim);
        using var norm = Norm(headsView, Config.Eps, weight);
        return norm.View(rows, heads * _headDim);
    }

    private Tensor Rope(Tensor input, int rows, int heads, int start)
    {
        var positions = new int[rows * heads];
        for (int i = 0; i < positions.Length; i++) positions[i] = start + i / heads;
        using var pos = CreateIntTensorOn(_allocator, positions, positions.Length);
        using var view = input.View(1, rows, heads, _headDim);
        using var rotated = Ops.RoPEEx(null, view, pos, _ropeDim, 0, 0,
            Config.RopeBase, 1f / Config.RopeScale, 0f, 1f, 0f, 0f);
        return rotated.View(rows, heads * _headDim);
    }

    private Tensor Attention(Tensor input, int layer, int rows, int start)
    {
        if (IsTensorParallel) return TensorParallelAttention(input, layer, rows, start);
        string p = _prefixes[layer];
        var projections = AttentionProjections(input, layer, start, null, out bool preparedQK);
        using var q0 = projections.Query;
        using var k0 = projections.Key;
        using var v = projections.Value;
        using var gate = projections.Gate;
        using var qn = preparedQK ? q0.CopyRef() : HeadNorm(q0, rows, Config.NumHeads, p + "attn_q_norm.weight");
        using var kn = preparedQK ? k0.CopyRef() : HeadNorm(k0, rows, Config.NumKVHeads, p + "attn_k_norm.weight");
        using var q = !preparedQK && _isSwa[layer] ? Rope(qn, rows, Config.NumHeads, start) : qn.CopyRef();
        using var k = !preparedQK && _isSwa[layer] ? Rope(kn, rows, Config.NumKVHeads, start) : kn.CopyRef();
        using var qHeads = ReshapeToHeads(q, Config.NumHeads, rows, _headDim);
        using var kHeads = ReshapeToHeads(k, Config.NumKVHeads, rows, _headDim);
        using var vHeads = ReshapeToHeads(v, Config.NumKVHeads, rows, _headDim);
        CopyToCache(_kvCacheK[layer], kHeads, start, rows);
        CopyToCache(_kvCacheV[layer], vHeads, start, rows);
        using var outputHeads = AttentionHeads(qHeads, layer, rows, start);
        using var flat = ReshapeFromHeads(outputHeads, Config.NumHeads, rows, _headDim);
        using var normed = HeadNormGate(flat, gate, rows);
        return LinearForward(normed, p + "attn_output.weight");
    }

    private unsafe Tensor AttentionHeads(Tensor q, int layer, int rows, int start)
    {
        int total = start + rows, window = _isSwa[layer] ? _slidingWindow : 0;
        var output = new Tensor(_allocator, DType.Float32, Config.NumHeads, rows, _headDim);
        float scale = 1f / MathF.Sqrt(_headDim);
        if (IsGgmlBackend && _kvCacheK[layer].ElementType == DType.Float16)
        {
            // Only the live window is uploaded on local-attention layers. The
            // earliest query in this chunk determines its first visible key.
            int firstKey = window > 0 ? Math.Max(0, start - window + 1) : 0;
            using var kWindow = _kvCacheK[layer].Narrow(1, firstKey, total - firstKey);
            using var vWindow = _kvCacheV[layer].Narrow(1, firstKey, total - firstKey);
            GgmlBasicOps.FusedPrefillAttentionF16KV(q, kWindow, vWindow, output,
                Config.NumHeads, Config.NumKVHeads, _headDim, rows, total - firstKey,
                (int)_kvCacheK[layer].Sizes[1], start - firstKey, window, scale, museSpark2DUploads: Use2DKvUploads);
            Record2DKvUpload(rows);
            InvalidateTensorDeviceCache(output);
            using (output)
            using (var flatOutput = output.View(rows, Config.NumHeads * _headDim))
                return ReshapeToHeads(flatOutput, Config.NumHeads, rows, _headDim);
        }
        // Compact GQA cache views. CUDA keeps F32 operands in the Spark-owned
        // path; other GGML backends retain their existing attention policy.
        using var k = ExpandKVHeads(_kvCacheK[layer], 1, total);
        using var v = ExpandKVHeads(_kvCacheV[layer], 1, total);
        if (IsGgmlBackend)
        {
            SparkPrefillAttentionF32(q, k, v, output, Config.NumHeads,
                Config.NumKVHeads, rows, total, start, window, scale);
            InvalidateTensorDeviceCache(output);
            // The native function consumes head-first Q but returns token-first
            // [tokens, heads * dim], irrespective of the output Tensor's shape.
            using (output)
            using (var flatOutput = output.View(rows, Config.NumHeads * _headDim))
                return ReshapeToHeads(flatOutput, Config.NumHeads, rows, _headDim);
        }
        float* qp = GetFloatPtr(q), kp = GetFloatPtr(k), vp = GetFloatPtr(v), op = GetFloatPtr(output);
        var scores = new float[total];
        int group = Config.NumHeads / Config.NumKVHeads;
        for (int head = 0; head < Config.NumHeads; head++)
        for (int row = 0; row < rows; row++)
        {
            int first = window > 0 ? Math.Max(0, start + row - window + 1) : 0;
            int end = start + row + 1;
            float max = float.NegativeInfinity;
            for (int key = first; key < end; key++)
            {
                float dot = 0;
                for (int d = 0; d < _headDim; d++) dot += qp[(head * rows + row) * _headDim + d] * kp[((head / group) * total + key) * _headDim + d];
                scores[key] = dot * scale;
                max = MathF.Max(max, scores[key]);
            }
            float sum = 0;
            for (int key = first; key < end; key++) { scores[key] = MathF.Exp(scores[key] - max); sum += scores[key]; }
            for (int d = 0; d < _headDim; d++)
            {
                float value = 0;
                for (int key = first; key < end; key++) value += scores[key] / sum * vp[((head / group) * total + key) * _headDim + d];
                op[(head * rows + row) * _headDim + d] = value;
            }
        }
        InvalidateTensorDeviceCache(output);
        return output;
    }

    // Prime one real prefill chunk; the generic 2048-token warmup otherwise
    // repeats the whole 816B decoder eight times with the default chunk size.
    protected override int GgmlPrefillChunkWarmupLength => _prefillChunk;
    protected override long KvCacheBytesPerToken => (long)Config.NumLayers * Config.NumKVHeads * _headDim * 2 * _kvCacheDtype.ToDType().Size();
    public override void PrepareForPrefill(int requiredContextTokens)
    {
        if (!UsesNativeExecutor) EnsureCacheCapacity(Math.Min(requiredContextTokens, _maxContextLength));
    }

    public override void WarmUpKernels()
    {
        if (UsesNativeExecutor) WarmUpNative();
        else base.WarmUpKernels();
    }

    private void EnsureCacheCapacity(int required)
    {
        if (required <= _cacheCapacity) return;
        if (required > _maxContextLength) throw new InvalidOperationException("Muse Spark context length exceeded.");
        int capacity = Math.Min(_maxContextLength, Math.Max(required, Math.Max(1, _cacheCapacity) * 2));
        for (int l = 0; l < Config.NumLayers; l++)
        {
            int previousRank = LayerSplitDegree > 1 ? GgmlBasicOps.GetActiveRank() : 0;
            bool previousPin = LayerSplitDegree > 1 && GgmlBasicOps.PinRank(LayerRank(l));
            Tensor k = null, v = null;
            try
            {
                k = new Tensor(_allocator, _kvCacheDtype.ToDType(), Config.NumKVHeads, capacity, _headDim);
                v = new Tensor(_allocator, _kvCacheDtype.ToDType(), Config.NumKVHeads, capacity, _headDim);
                InitializeCacheTensor(k); InitializeCacheTensor(v);
                if (_cacheSeqLen > 0)
                {
                    CopyCacheRows(_kvCacheK[l], k, _cacheSeqLen);
                    CopyCacheRows(_kvCacheV[l], v, _cacheSeqLen);
                }
                if (_kvCacheK[l] != null) InvalidateTensorDeviceCache(_kvCacheK[l]);
                if (_kvCacheV[l] != null) InvalidateTensorDeviceCache(_kvCacheV[l]);
                InvalidateTensorDeviceCache(k); InvalidateTensorDeviceCache(v);
                var oldK = _kvCacheK[l]; var oldV = _kvCacheV[l];
                _kvCacheK[l] = k; _kvCacheV[l] = v;
                k = v = null; // The model now owns this complete layer pair.
                oldK?.Dispose(); oldV?.Dispose();
            }
            finally
            {
                // A later allocation may fail after earlier layers grew. They
                // remain valid at their own physical strides; discard only the
                // unfinished pair and leave the common capacity at its old floor.
                k?.Dispose(); v?.Dispose();
                if (LayerSplitDegree > 1) GgmlBasicOps.RestoreRankPin(previousPin, previousRank);
            }
        }
        _cacheCapacity = capacity;
    }

    protected override void ResetKVCacheCore()
    {
        if (UsesNativeExecutor) ResetNative();
        _cacheSeqLen = 0;
        ClearPendingVisionEmbeddings();
    }
    protected override void TruncateKVCacheCore(int tokenCount)
    {
        if (UsesNativeExecutor)
        {
            if (!RewindNative(tokenCount))
                throw new InvalidOperationException($"Muse Spark cannot rewind its sliding-window cache to {tokenCount} tokens " +
                    $"(head at {_cacheSeqLen}); use TryTruncateKVCache and re-prefill when it declines.");
            ClearPendingVisionEmbeddings();
            return;
        }
        if ((uint)tokenCount > (uint)_cacheSeqLen) throw new ArgumentOutOfRangeException(nameof(tokenCount));
        _cacheSeqLen = tokenCount;
        ClearPendingVisionEmbeddings();
    }
    protected override bool TryTruncateKVCacheCore(int tokenCount)
    {
        if (UsesNativeExecutor)
        {
            if (!RewindNative(tokenCount)) return false;
            ClearPendingVisionEmbeddings();
            return true;
        }
        if (!CanTruncateKVCache(_cacheSeqLen, tokenCount)) return false;
        TruncateKVCacheCore(tokenCount);
        return true;
    }
    public override bool SupportsKVStateSnapshot => !UsesNativeExecutor && _kvCacheK != null;
    public override string KVStateFingerprint => $"muse-spark|layers={Config.NumLayers}|heads={Config.NumHeads}:{Config.NumKVHeads}:{_headDim}" +
        $"|swa={_slidingWindow}:{string.Join(',', _isSwa)}|rope={_ropeDim}:{Config.RopeBase}:{Config.RopeScale}" +
        $"|dtype={(UsesNativeExecutor ? "f16" : _kvCacheDtype.ToShortString())}|exec={(UsesNativeExecutor ? "native" : "managed")}";
    public override long ComputeKVBlockByteSize(int tokens)
        => UsesNativeExecutor ? 0 : KvBlockTransfer.ComputeBlockByteSize(_kvCacheK, _kvCacheV, tokens);
    public override bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        => !UsesNativeExecutor && KvBlockTransfer.Extract(_allocator, _kvCacheK, _kvCacheV, _cacheSeqLen, startToken, tokenCount, destination);
    public override bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
    {
        if (UsesNativeExecutor) return false;
        if (destToken < 0 || tokenCount <= 0 || (long)destToken + tokenCount > _maxContextLength) return false;
        EnsureCacheCapacity(destToken + tokenCount);
        if (!KvBlockTransfer.Inject(_allocator, _kvCacheK, _kvCacheV, _cacheSeqLen, destToken, tokenCount, source)) return false;
        _cacheSeqLen = destToken + tokenCount;
        InvalidateKvDeviceCopies();
        return true;
    }

    private void InvalidateKvDeviceCopies()
    {
        if (_kvCacheK == null) return;
        if (IsTensorParallel && _tpKvFirst != null)
        {
            _tpGroup.RunPerRank(rank =>
            {
                for (int layer = 0; layer < _kvCacheK.Length; layer++)
                {
                    if (_kvCacheK[layer] == null) continue;
                    using var k = _kvCacheK[layer].Narrow(0, _tpKvFirst[rank], _tpKvCount[rank]);
                    using var v = _kvCacheV[layer].Narrow(0, _tpKvFirst[rank], _tpKvCount[rank]);
                    InvalidateTensorDeviceCache(k); InvalidateTensorDeviceCache(v);
                }
            });
            return;
        }
        for (int l = 0; l < _kvCacheK.Length; l++)
        {
            int previousRank = LayerSplitDegree > 1 ? GgmlBasicOps.GetActiveRank() : 0;
            bool previousPin = LayerSplitDegree > 1 && GgmlBasicOps.PinRank(LayerRank(l));
            try
            {
                if (_kvCacheK[l] != null) InvalidateTensorDeviceCache(_kvCacheK[l]);
                if (_kvCacheV[l] != null) InvalidateTensorDeviceCache(_kvCacheV[l]);
            }
            finally
            {
                if (LayerSplitDegree > 1) GgmlBasicOps.RestoreRankPin(previousPin, previousRank);
            }
        }
    }

    public override void Dispose()
    {
        DisposeNative();
        DisposeVisionState();
        foreach (var t in _ones.Values) t.Dispose();
        _ones.Clear();
        InvalidateKvDeviceCopies();
        if (_kvCacheK != null) foreach (var t in _kvCacheK) t?.Dispose();
        if (_kvCacheV != null) foreach (var t in _kvCacheV) t?.Dispose();
        _kvCacheK = _kvCacheV = null;
        _pagedKeys = _pagedValues = null;
        _pagedBlocks = _pagedBlockSize = 0;
        if (IsGgmlBackend) OnBeforeReleaseGgmlDeviceResidency();
        DisposeSparkTensorParallel();
        base.Dispose();
    }
}
