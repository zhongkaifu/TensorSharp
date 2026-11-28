// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.IO;
using System.Linq;
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models;

/// <summary>
/// MiniMax-M3.1 GGUF decoder (architecture <c>minimax-m3</c>). GQA attention with per-head
/// Q/K RMSNorm and partial NeoX RoPE, MiniMax Sparse Attention on every layer that carries an
/// indexer (a per-KV-group top-k over fixed-size blocks of absolute positions), leading dense
/// SwiGLU-OAI layers, then a sigmoid-routed MoE whose NVFP4 experts carry per-expert output
/// scales, plus one shared expert. The GGUF norms already hold Gemma's 1+w.
///
/// <para>The model runs only on the native whole-model executor
/// (<c>ggml_ops_minimax_m3_exec.cpp</c>); this class parses and validates the checkpoint,
/// owns the tokenizer and serves the host contracts (per-request slots, batched decode and
/// prefill, prefix reuse, speculation) over the executor's slots.</para>
/// </summary>
public partial class MiniMaxM3Model : ModelBase
{
    /// <summary>The GGUF <c>general.architecture</c> value and protocol key.</summary>
    internal const string ArchitectureId = "minimax-m3";

    private readonly int _headDim, _ropeDim, _denseLead;
    private readonly int _numExperts, _numExpertsUsed, _expertFfnLength, _numSharedExperts;
    private readonly float _expertWeightsScale;
    private readonly bool _expertWeightsNorm;
    private readonly int _indexerHeads, _indexerDim, _indexerTopK, _indexerBlock, _indexerLocalBlocks;
    private readonly bool[] _isMsaLayer;
    private readonly int _scaledExpertLayers;
    private readonly string _msaMode;

    public MiniMaxM3Model(string ggufPath, BackendType backend, int tpDegree = 1,
        ITensorParallelGroup tpGroup = null, int layerSplitDegree = 1, string draftModelPath = null)
        : this(ggufPath, backend, tpDegree, tpGroup, layerSplitDegree, draftModelPath, loadExecutor: true)
    {
    }

    // loadExecutor=false parses and validates a checkpoint without the native executor, for tests;
    // such an instance refuses every forward.
    internal MiniMaxM3Model(string ggufPath, BackendType backend, int tpDegree, ITensorParallelGroup tpGroup,
        int layerSplitDegree, string draftModelPath, bool loadExecutor)
        // The native executor owns every GPU it uses; the managed base keeps only
        // device 0's context (vision tower, tokenizer-side helpers).
        : base(ggufPath, loadExecutor ? ValidateConstruction(backend, tpDegree, tpGroup, layerSplitDegree, draftModelPath) : backend,
            1, null, 1)
    {
        try
        {
            string arch = _gguf.GetString("general.architecture") ?? ArchitectureId;
            Config = new ModelConfig { Architecture = ArchitectureId };
            MetadataArchitecture = arch;
            ParseBaseConfig();
            _headDim = Config.HeadDim;
            _ropeDim = (int)_gguf.GetUint32($"{arch}.rope.dimension_count", (uint)_headDim);
            _denseLead = (int)_gguf.GetUint32($"{arch}.leading_dense_block_count", 0);
            _numExperts = Config.NumExperts = (int)_gguf.GetUint32($"{arch}.expert_count");
            _numExpertsUsed = Config.NumExpertsUsed = (int)_gguf.GetUint32($"{arch}.expert_used_count");
            _expertFfnLength = (int)_gguf.GetUint32($"{arch}.expert_feed_forward_length");
            _numSharedExperts = (int)_gguf.GetUint32($"{arch}.expert_shared_count", 0);
            _expertWeightsScale = _gguf.GetFloat32($"{arch}.expert_weights_scale", 1f);
            _expertWeightsNorm = _gguf.GetBool($"{arch}.expert_weights_norm", false);
            _indexerHeads = (int)_gguf.GetUint32($"{arch}.attention.indexer.head_count");
            _indexerDim = (int)_gguf.GetUint32($"{arch}.attention.indexer.key_length");
            _indexerTopK = (int)_gguf.GetUint32($"{arch}.attention.indexer.top_k");
            _indexerBlock = (int)_gguf.GetUint32($"{arch}.attention.indexer.block_size");
            _indexerLocalBlocks = (int)_gguf.GetUint32($"{arch}.attention.indexer.local_blocks", 1);
            ValidateGeometry(arch);
            ParseTokenizer();
            _isMsaLayer = new bool[Config.NumLayers];
            _scaledExpertLayers = ValidateWeights();
            _msaMode = Environment.GetEnvironmentVariable(MiniMaxM3Architecture.MsaModeVariable)?.Trim() is { Length: > 0 } mode
                ? mode : "auto";
            // Every executor path keeps K, V and the index keys in F16; report what runs.
            if (KvCacheDtypeConfig.IsExplicitlySet && KvCacheDtypeConfig.Current == KvCacheDtype.F32 && loadExecutor)
                Console.Error.WriteLine("[minimax-m3] KV_CACHE_DTYPE=f32 requested, but the native executor keeps K, V " +
                    "and index keys in F16; running with kvCacheDtype=f16.");
            _kvCacheDtype = KvCacheDtype.F16;
            if (!loadExecutor)
            {
                _maxContextLength = ResolveConfiguredContextLength();
                return;
            }
            if (backend == BackendType.GgmlCpu)
                Console.Error.WriteLine("[minimax-m3] --backend ggml_cpu: the native executor runs on ONE CPU device. " +
                    "That is a correctness path, not a serving path. Pass --backend ggml_cuda for the GPU executor " +
                    "(ggml_cpu is also the server default when --backend is omitted off macOS).");
            PrepareDspark(draftModelPath);
            InitNativeExecutor(ggufPath, backend, layerSplitDegree, tpDegree);
            Console.WriteLine($"MiniMax-M3.1: {Config.NumLayers} layers ({_denseLead} dense, " +
                $"{_isMsaLayer.Count(m => m)} with sparse attention), {_numExperts} experts/top-{_numExpertsUsed} " +
                $"+ {_numSharedExperts} shared, indexer {_indexerHeads}x{_indexerDim} top-{_indexerTopK} of " +
                $"{_indexerBlock}-row blocks, attention {_msaMode}, {_backend}.");
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>The backend and placement checks a load runs before the base constructor maps anything.</summary>
    private static BackendType ValidateConstruction(BackendType backend, int tpDegree, ITensorParallelGroup tpGroup,
        int layerSplitDegree, string draftModelPath)
    {
        MiniMaxM3Architecture.ValidateLoad(backend, tpGroup, draftModelPath);
        if (tpDegree > 1)
        {
            if (backend != BackendType.GgmlCuda)
                throw new NotSupportedException(MiniMaxM3Architecture.TensorParallelBackendRefusal);
            if (layerSplitDegree > 1)
                throw new NotSupportedException("--layer-split and --tp select different execution modes and cannot be combined.");
            if (!string.IsNullOrWhiteSpace(draftModelPath))
                throw new NotSupportedException(MiniMaxM3Architecture.TensorParallelDraftRefusal);
        }
        if (layerSplitDegree > 1 && backend != BackendType.GgmlCuda)
            throw new NotSupportedException("MiniMax-M3.1 layer placement (--layer-split) requires --backend ggml_cuda.");
        return backend;
    }

    private void ValidateGeometry(string arch)
    {
        bool hasMoe = _denseLead < Config.NumLayers;
        if (Config.NumLayers <= 0 || Config.HiddenSize <= 0 || Config.NumHeads <= 0 || Config.NumKVHeads <= 0 ||
            Config.NumHeads % Config.NumKVHeads != 0 || _headDim <= 0 ||
            (Config.ValueLength > 0 && Config.ValueLength != _headDim) ||
            _ropeDim <= 0 || _ropeDim > _headDim || _ropeDim % 2 != 0 ||
            !(Config.Eps > 0) || !(Config.RopeBase > 0) || !(Config.RopeScale > 0) ||
            _denseLead < 0 || _denseLead > Config.NumLayers || (_denseLead > 0 && Config.IntermediateSize <= 0))
            throw new InvalidDataException("Invalid MiniMax-M3 attention, norm or dense-layer geometry.");
        if (hasMoe && (_numExperts <= 0 || _numExpertsUsed <= 0 || _numExpertsUsed > _numExperts ||
            _expertFfnLength <= 0 || _numSharedExperts < 0 || !float.IsFinite(_expertWeightsScale) || !(_expertWeightsScale > 0)))
            throw new InvalidDataException("Invalid MiniMax-M3 expert geometry.");
        if (hasMoe && _gguf.GetUint32($"{arch}.expert_gating_func", 0) != 2)
            throw new NotSupportedException("MiniMax-M3 requires sigmoid expert routing (expert_gating_func = 2).");
        // Index head h selects the blocks KV head h (query heads h*G..h*G+G-1) attends to, and the
        // index queries and keys take the same partial RoPE as the main ones.
        if (_indexerHeads != Config.NumKVHeads || _indexerDim <= 0 || _ropeDim > _indexerDim ||
            _indexerTopK <= 0 || _indexerBlock <= 0 || _indexerLocalBlocks < 0 || _indexerLocalBlocks > _indexerTopK)
            throw new InvalidDataException(
                $"Invalid MiniMax-M3 sparse-attention indexer geometry: {_indexerHeads} index heads of {_indexerDim} " +
                $"(the model has {Config.NumKVHeads} KV heads), top_k {_indexerTopK}, block {_indexerBlock}, " +
                $"local blocks {_indexerLocalBlocks}.");
    }

    // Do not prefault hundreds of gigabytes of sparse offloaded experts: the OS
    // demand-pages selected expert rows from the unchanged GGUF mmap.
    protected override bool ShouldPrefaultWeight(GgufTensorInfo info)
        => !MoeOffload.IsOffloadedExpertWeightName(info.Name);

    // The executor keeps K, V and index keys as F16 rows that its kernels read directly.
    protected override bool SupportsBlockQuantizedKvCache => false;

    /// <summary>
    /// Check every tensor the executor reads against the metadata (shapes in GGUF order, ne0
    /// first). The indexer is required on the MoE layers and optional, all four tensors or
    /// none, on the leading dense layers (MiniMax-M3 has none there, M3.1 has them on every
    /// layer); a layer is sparse iff it carries one. The per-expert <c>.scale</c> vectors the
    /// NVFP4 conversion writes are optional as in llama.cpp, but must cover every expert.
    /// Returns the number of MoE layers whose experts carry such scales.
    /// </summary>
    private int ValidateWeights()
    {
        bool Has(string name) => _gguf.Tensors.ContainsKey(name);
        void Require(string name, params ulong[] shape)
        {
            if (!_gguf.Tensors.TryGetValue(name, out var tensor) || !tensor.Shape.AsSpan().SequenceEqual(shape))
                throw new InvalidDataException($"MiniMax-M3 tensor {name} is missing or has incompatible dimensions " +
                    $"(expected [{string.Join(", ", shape)}]).");
        }
        ulong h = (ulong)Config.HiddenSize, hd = (ulong)_headDim, vocab = (ulong)Config.VocabSize;
        ulong q = (ulong)(Config.NumHeads * _headDim), kv = (ulong)(Config.NumKVHeads * _headDim);
        ulong idx = (ulong)_indexerDim, experts = (ulong)_numExperts, eff = (ulong)_expertFfnLength;
        Require("token_embd.weight", h, vocab);
        Require("output.weight", h, vocab);
        Require("output_norm.weight", h);
        int scaled = 0;
        for (int l = 0; l < Config.NumLayers; l++)
        {
            string p = $"blk.{l}.";
            bool dense = l < _denseLead;
            Require(p + "attn_norm.weight", h);
            Require(p + "ffn_norm.weight", h);
            Require(p + "attn_q.weight", h, q);
            Require(p + "attn_k.weight", h, kv);
            Require(p + "attn_v.weight", h, kv);
            Require(p + "attn_output.weight", q, h);
            Require(p + "attn_q_norm.weight", hd);
            Require(p + "attn_k_norm.weight", hd);

            string[] indexer = { "indexer.q_proj.weight", "indexer.k_proj.weight", "indexer.q_norm.weight", "indexer.k_norm.weight" };
            int present = indexer.Count(n => Has(p + n));
            if (present != 0 && present != indexer.Length)
                throw new InvalidDataException($"MiniMax-M3 layer {l} carries {present} of the {indexer.Length} indexer tensors.");
            if (present == 0 && !dense)
                throw new InvalidDataException($"MiniMax-M3 MoE layer {l} has no sparse-attention indexer (blk.{l}.indexer.*).");
            _isMsaLayer[l] = present != 0;
            if (_isMsaLayer[l])
            {
                Require(p + "indexer.q_proj.weight", h, (ulong)_indexerHeads * idx);
                Require(p + "indexer.k_proj.weight", h, idx);
                Require(p + "indexer.q_norm.weight", idx);
                Require(p + "indexer.k_norm.weight", idx);
            }

            if (dense)
            {
                ulong ff = (ulong)Config.IntermediateSize;
                Require(p + "ffn_gate.weight", h, ff);
                Require(p + "ffn_up.weight", h, ff);
                Require(p + "ffn_down.weight", ff, h);
                continue;
            }
            Require(p + "ffn_gate_inp.weight", h, experts);
            Require(p + "exp_probs_b.bias", experts);
            // The selection bias sits around 16..37 while the sigmoid scores it is added to lie in
            // (0, 1): stored at F16 resolution (1/64 there) it would already pick other experts.
            if (_gguf.Tensors[p + "exp_probs_b.bias"].Type != GgmlTensorType.F32)
                throw new InvalidDataException($"MiniMax-M3 tensor {p}exp_probs_b.bias must be F32: the routing bias " +
                    "decides expert selection at a resolution F16 and BF16 cannot hold.");
            Require(p + "ffn_gate_exps.weight", h, eff, experts);
            Require(p + "ffn_up_exps.weight", h, eff, experts);
            Require(p + "ffn_down_exps.weight", eff, h, experts);
            int scales = 0;
            foreach (string e in new[] { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" })
            {
                if (!_gguf.Tensors.TryGetValue(p + e + ".scale", out var scale)) continue;
                if (!scale.Shape.AsSpan().SequenceEqual(new[] { experts }) || scale.Type != GgmlTensorType.F32)
                    throw new InvalidDataException($"MiniMax-M3 tensor {p}{e}.scale must be F32 [{experts}], one output " +
                        "scale per expert.");
                scales++;
            }
            if (scales > 0) scaled++;
            if (_numSharedExperts > 0)
            {
                ulong shared = (ulong)(_expertFfnLength * _numSharedExperts);
                Require(p + "ffn_gate_shexp.weight", h, shared);
                Require(p + "ffn_up_shexp.weight", h, shared);
                Require(p + "ffn_down_shexp.weight", shared, h);
            }
        }
        return scaled;
    }

    protected override float[] ForwardCore(int[] tokens)
    {
        if (tokens == null || tokens.Length == 0) throw new ArgumentException("A forward pass needs at least one token.", nameof(tokens));
        if (!UsesNativeExecutor)
            throw new InvalidOperationException("MiniMax-M3.1 runs only on its native executor, which this instance did not load.");
        return ForwardNative(tokens);
    }

    // K and V (n_kv_heads x head_dim each) plus one index key per layer, all F16.
    protected override long KvCacheBytesPerToken
        => (long)Config.NumLayers * (2L * Config.NumKVHeads * _headDim + _indexerDim) * sizeof(ushort);

    public override void WarmUpKernels()
    {
        if (UsesNativeExecutor) WarmUpNative();
    }

    protected override void ResetKVCacheCore()
    {
        if (UsesNativeExecutor) ResetNative();
        _cacheSeqLen = 0;
        ClearPendingVisionRows();
    }

    // Every layer keeps every row (no sliding window, no recurrent state) and the index keys
    // advance with K and V, so a rewind to any token is exact.
    protected override void TruncateKVCacheCore(int tokenCount)
    {
        if (!TryTruncateKVCacheCore(tokenCount))
            throw new InvalidOperationException($"MiniMax-M3.1 cannot rewind its cache to {tokenCount} tokens " +
                $"(head at {_cacheSeqLen}).");
    }

    protected override bool TryTruncateKVCacheCore(int tokenCount)
    {
        if (!UsesNativeExecutor)
        {
            if ((uint)tokenCount > (uint)_cacheSeqLen) return false;
            _cacheSeqLen = tokenCount;
        }
        else if (!RewindNative(tokenCount))
            return false;
        ClearPendingVisionRows();
        return true;
    }

    /// <summary>
    /// The cache namespace: geometry, the sparse-attention indexer and which layers use it, and
    /// whether attention was forced dense (TS_MINIMAX_M3_MSA=dense changes every K/V row past
    /// top_k x block_size, so its caches are never shared with the sparse model's).
    /// </summary>
    public override string KVStateFingerprint =>
        $"minimax-m3|layers={Config.NumLayers}|heads={Config.NumHeads}:{Config.NumKVHeads}:{_headDim}" +
        $"|rope={_ropeDim}:{Config.RopeBase}:{Config.RopeScale}" +
        $"|msa={_indexerHeads}:{_indexerDim}:{_indexerTopK}:{_indexerBlock}:{_indexerLocalBlocks}" +
        $":{string.Concat(_isMsaLayer.Select(m => m ? '1' : '0'))}" +
        $"|attn={(string.Equals(_msaMode, "dense", StringComparison.Ordinal) ? "dense" : "sparse")}" +
        $"|dtype=f16|exec={(UsesNativeExecutor ? "native" : "none")}";

    public override void Dispose()
    {
        DisposeNative();
        ClearPendingVisionRows();
        DisposeVisionEncoder();
        base.Dispose();
    }
}
