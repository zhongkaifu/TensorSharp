// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.MLX;

namespace TensorSharp.Models
{
    /// <summary>
    /// GPT OSS (Mixture-of-Experts) transformer model.
    /// Key features:
    ///   - MoE FFN with TopK routing + softmax on selected experts
    ///   - Alternating SWA (even layers) / full causal (odd layers) attention
    ///   - Attention sinks for SWA layers
    ///   - SiLU with alpha scaling and clamping (SiLUAlphaLimit)
    ///   - RoPE NeoX with yarn scaling
    ///   - Bias on all attention and FFN projections
    /// Optimizations:
    ///   - Fused QKV projection (3 matmuls -> 1)
    ///   - Expert batching in MoE (N*K matmuls -> up to numExperts batched matmuls)
    ///   - Pre-computed weight name strings (zero allocation per forward)
    ///   - Cached attention sinks arrays
    ///   - SIMD-vectorized bias addition and activation
    /// </summary>
    public partial class GptOssModel : ModelBase
    {
        // Bound the MLX lazy-graph depth across the per-layer dispatch loop.
        // Override via TS_MLX_EVAL_EVERY_N_LAYERS. GptOss has 24 layers; eval=16
        // means one boundary at layer 16.
        private static readonly int MlxEvalEveryNLayers = ResolveMlxEvalEveryNLayers();
        private static int ResolveMlxEvalEveryNLayers()
        {
            string env = Environment.GetEnvironmentVariable("TS_MLX_EVAL_EVERY_N_LAYERS");
            if (!string.IsNullOrWhiteSpace(env) && int.TryParse(env, out int v) && v > 0)
                return v;
            return 16;
        }

        // Minimum total sequence length to use the MLX device-side decode
        // attention with sinks. Default 1 (always-on). Empirically the
        // device-side kernel beats the host SIMD CPU path even at short
        // kvLen (~300) on M-series — measured +23% decode tok/s on
        // gpt-oss-20B Q8_0 — and the win grows with kvLen since the host
        // path scales linearly with kvLen on the multi-GB cache download.
        // Override via TS_MLX_SINKS_ATTN_MIN_KV_LEN if a workload regresses.
        private static readonly int MlxSinksAttnMinKvLen = ResolveMlxSinksAttnMinKvLen();
        private static int ResolveMlxSinksAttnMinKvLen()
        {
            string env = Environment.GetEnvironmentVariable("TS_MLX_SINKS_ATTN_MIN_KV_LEN");
            if (!string.IsNullOrWhiteSpace(env) && int.TryParse(env, out int v) && v > 0)
                return v;
            return 1;
        }

        private Tensor[] _kvCacheK;
        private Tensor[] _kvCacheV;
        private int _numExperts;
        private int _numExpertsUsed;
        private int _slidingWindow;
        private int _expertFfnLength;

        private const float SiluAlpha = 1.702f;
        private const float SiluLimit = 7.0f;

        private string[][] _layerNames;
        private string[][][] _expertNames;
        private float[][] _layerSinks;
        // Per-layer MLX-backed 1D Float32 [numHeads] tensor mirror of
        // _layerSinks for the device-side decode path. Lazily populated on
        // first use; reused across decode calls.
        private Tensor[] _layerSinksMlx;
        private int _qDim, _kDim;
        private bool _isQkvFused;

        private int[] _moeExpertCounts;
        private int[] _moeExpertOffsets;
        private int[] _moeTokenMap;
        private float[] _moeWeightMap;

        // Pooled per-call scratch for MoE routing. Reallocated lazily when the
        // current request needs a larger seqLen (prefill). Decode reuses the
        // same buffers across all layers/steps, eliminating the per-call
        // float[]/int[] allocation in MoERoute().
        private float[] _moeRoutingWeightsScratch;
        private int[] _moeSelectedExpertsScratch;
        private int[] _moeTopKScratch;

        // Per-layer stacked-along-experts views into the original `ffn_gate_exps.weight`,
        // `ffn_up_exps.weight`, `ffn_down_exps.weight` 3D blocks (loaded into
        // `ModelBase._stackedExpertWeights`). Used by the fused MoE prefill kernel
        // (TryMoEPrefillFused) to dispatch one ggml_cgraph per layer (with
        // ggml_mul_mat_id + ggml_add_id + swiglu_oai) instead of looping over
        // active experts per token. FuseExpertGateUpWeights leaves these
        // per-expert SLICED `_quantWeights` entries gone but the underlying
        // 3D stacked storage is untouched (FuseExpertGateUpWeights only
        // disposes the per-expert *views*, not the bulk buffer).
        private StackedExpertWeights[] _layerStackedGate;
        private StackedExpertWeights[] _layerStackedUp;
        private StackedExpertWeights[] _layerStackedDown;

        // Per-layer stacked biases for the fused MoE prefill kernel. Layout is
        // contiguous [bias_dim, num_experts] f32 so that the kernel can hand
        // them directly to ggml_add_id (which expects ne0=bias_dim, ne1=num_experts).
        // Built once at init time from `ffn_gate_up_exps.{e}.bias` (already
        // gate || up concatenated by FuseExpertGateUpWeights, size 2*n_ff) and
        // `ffn_down_exps.{e}.bias` (hidden_dim).
        private float[][] _layerGateUpBiasStacked;  // shape [2*n_ff * num_experts] per layer
        private float[][] _layerDownBiasStacked;    // shape [hidden_dim * num_experts] per layer
        private int _layerStackedReady;             // 1 once InitMoeStackedWeights has run

        public GptOssModel(string ggufPath, BackendType backend)
            : base(ggufPath, backend)
        {
            string arch = _gguf.GetString("general.architecture") ?? "gpt-oss";
            Config = new ModelConfig { Architecture = arch };
            ParseBaseConfig();

            _numExperts = (int)_gguf.GetUint32($"{arch}.expert_count", 0);
            _numExpertsUsed = (int)_gguf.GetUint32($"{arch}.expert_used_count", 0);
            _slidingWindow = (int)_gguf.GetUint32($"{arch}.attention.sliding_window", 128);
            _expertFfnLength = (int)_gguf.GetUint32($"{arch}.expert_feed_forward_length", 0);

            Config.NumExperts = _numExperts;
            Config.NumExpertsUsed = _numExpertsUsed;
            Config.SlidingWindow = _slidingWindow;
            Config.OriginalContextLength = (int)_gguf.GetUint32($"{arch}.rope.scaling.original_context_length", 4096);

            ParseTokenizer();

            Console.WriteLine($"Model: {arch}, Layers={Config.NumLayers}, Hidden={Config.HiddenSize}, " +
                $"Heads={Config.NumHeads}, KVHeads={Config.NumKVHeads}, HeadDim={Config.HeadDim}, Vocab={Config.VocabSize}");
            Console.WriteLine($"RoPE base={Config.RopeBase}, scale={Config.RopeScale}, eps={Config.Eps}");
            Console.WriteLine($"MoE: {_numExperts} experts, {_numExpertsUsed} used, " +
                $"SlidingWindow={_slidingWindow}, ExpertFFN={_expertFfnLength}");

            LoadWeights();
            SplitExpertBiases();
            // Snapshot the gate/up biases per expert BEFORE FuseExpertGateUpWeights
            // disposes them — we need them in their original split shape to build
            // the stacked-by-expert bias tables for the fused MoE prefill kernel.
            float[][] preFuseGateBias = SnapshotPerExpertBiases("ffn_gate_exps", _expertFfnLength);
            float[][] preFuseUpBias = SnapshotPerExpertBiases("ffn_up_exps", _expertFfnLength);
            FuseExpertGateUpWeights();
            FuseQKVWeights();
            PrepareCudaQuantizedWeightsForInference();
            int maxContextLength = ResolveConfiguredContextLength();
            int initialCacheLength = ResolveInitialCacheAllocationLength(maxContextLength);
            if (initialCacheLength < maxContextLength)
                Console.WriteLine($"Initial {_backend} KV cache allocation: {initialCacheLength} tokens (grows on demand up to {maxContextLength}).");
            InitKVCache(initialCacheLength, maxContextLength);
            PrecomputeConstants();
            InitMoeStackedWeights(preFuseGateBias, preFuseUpBias);
        }

        // Build a per-(layer,expert) snapshot of bias arrays before FuseExpertGateUpWeights
        // collapses them. Returns float[layer][expert*biasDim + d]. Caller is
        // responsible for dimension consistency. Returns null if no biases found
        // for the first layer (some MoE models don't ship gate/up biases).
        private float[][] SnapshotPerExpertBiases(string kind, int biasDim)
        {
            int numLayers = Config.NumLayers;
            float[][] result = new float[numLayers][];
            bool any = false;
            for (int l = 0; l < numLayers; l++)
            {
                float[] perLayer = new float[biasDim * _numExperts];
                bool layerHasAny = false;
                for (int e = 0; e < _numExperts; e++)
                {
                    string biasName = $"blk.{l}.{kind}.{e}.bias";
                    if (_weights.TryGetValue(biasName, out var biasTensor) && biasTensor != null)
                    {
                        float[] biasData = TensorToFloatArray(biasTensor);
                        int copyLen = Math.Min(biasData.Length, biasDim);
                        Array.Copy(biasData, 0, perLayer, e * biasDim, copyLen);
                        layerHasAny = true;
                    }
                }
                result[l] = layerHasAny ? perLayer : null;
                any |= layerHasAny;
            }
            return any ? result : null;
        }

        // Build per-layer stacked weight + bias views for the fused MoE prefill
        // kernel (TryMoEPrefillFused). Stacked weights are zero-cost views into
        // the original 3D `_exps.weight` blocks loaded by ModelBase. Stacked
        // biases are small contiguous f32 arrays built once from the per-expert
        // biases captured prior to FuseExpertGateUpWeights.
        private unsafe void InitMoeStackedWeights(float[][] preFuseGateBias, float[][] preFuseUpBias)
        {
            int numLayers = Config.NumLayers;
            int hidden = Config.HiddenSize;
            int nFf = _expertFfnLength;

            _layerStackedGate = new StackedExpertWeights[numLayers];
            _layerStackedUp = new StackedExpertWeights[numLayers];
            _layerStackedDown = new StackedExpertWeights[numLayers];
            _layerGateUpBiasStacked = new float[numLayers][];
            _layerDownBiasStacked = new float[numLayers][];

            int gotWeights = 0;
            int gotBiases = 0;
            for (int l = 0; l < numLayers; l++)
            {
                string p = $"blk.{l}.";
                _stackedExpertWeights.TryGetValue(p + "ffn_gate_exps.weight", out _layerStackedGate[l]);
                _stackedExpertWeights.TryGetValue(p + "ffn_up_exps.weight", out _layerStackedUp[l]);
                _stackedExpertWeights.TryGetValue(p + "ffn_down_exps.weight", out _layerStackedDown[l]);
                if (_layerStackedGate[l] != null && _layerStackedUp[l] != null && _layerStackedDown[l] != null)
                    gotWeights++;

                if (preFuseGateBias != null && preFuseUpBias != null
                    && preFuseGateBias[l] != null && preFuseUpBias[l] != null)
                {
                    // Stack gate || up bias per expert into a contiguous
                    // [2*n_ff, num_experts] f32 array (gate first n_ff, then up).
                    // ggml_add_id reads bias[d, ids[u, t]] so layout must match
                    // expert e occupying offset e * (2*n_ff).
                    float[] fused = new float[2 * nFf * _numExperts];
                    for (int e = 0; e < _numExperts; e++)
                    {
                        int dst = e * 2 * nFf;
                        Array.Copy(preFuseGateBias[l], e * nFf, fused, dst, nFf);
                        Array.Copy(preFuseUpBias[l], e * nFf, fused, dst + nFf, nFf);
                    }
                    _layerGateUpBiasStacked[l] = fused;
                    gotBiases++;
                }

                // Down biases live in `_weights[blk.{l}.ffn_down_exps.{e}.bias]`
                // as f32 [1, hidden_dim]. Stack them across experts for the kernel.
                bool hasDownBias = false;
                float[] downStack = new float[hidden * _numExperts];
                for (int e = 0; e < _numExperts; e++)
                {
                    string downBiasName = $"blk.{l}.ffn_down_exps.{e}.bias";
                    if (_weights.TryGetValue(downBiasName, out var bt) && bt != null)
                    {
                        float[] bd = TensorToFloatArray(bt);
                        Array.Copy(bd, 0, downStack, e * hidden, Math.Min(bd.Length, hidden));
                        hasDownBias = true;
                    }
                }
                if (hasDownBias)
                    _layerDownBiasStacked[l] = downStack;
            }

            _layerStackedReady = (gotWeights == numLayers) ? 1 : 0;
            if (gotWeights > 0)
            {
                Console.WriteLine($"  Fused MoE prefill: stacked weights ready for {gotWeights}/{numLayers} layers, " +
                    $"stacked gate/up biases for {gotBiases}/{numLayers} layers");
            }
        }

        #region Weight Fusion and Pre-computation

        private void SplitExpertBiases()
        {
            int split = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                foreach (string kind in new[] { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" })
                {
                    string biasName = $"blk.{l}.{kind}.bias";
                    if (!_weights.TryGetValue(biasName, out var biasTensor))
                        continue;

                    int numExp = (int)biasTensor.Sizes[0];
                    int biasDim = (int)biasTensor.Sizes[1];
                    float[] biasData = TensorToFloatArray(biasTensor);

                    for (int e = 0; e < numExp; e++)
                    {
                        float[] expertBias = new float[biasDim];
                        for (int d = 0; d < biasDim; d++)
                            expertBias[d] = biasData[e * biasDim + d];
                        _weights[$"blk.{l}.{kind}.{e}.bias"] = CreateFloatTensor(expertBias, 1, biasDim);
                    }
                    _weights.Remove(biasName);
                    biasTensor.Dispose();
                    split++;
                }
            }
            if (split > 0)
                Console.WriteLine($"  Split expert biases: {split} tensors");
        }

        private unsafe void FuseExpertGateUpWeights()
        {
            int fused = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                for (int e = 0; e < _numExperts; e++)
                {
                    string gateName = $"blk.{l}.ffn_gate_exps.{e}.weight";
                    string upName = $"blk.{l}.ffn_up_exps.{e}.weight";
                    string fusedName = $"blk.{l}.ffn_gate_up_exps.{e}.weight";

                    if (_quantWeights.TryGetValue(gateName, out var gw) &&
                        _quantWeights.TryGetValue(upName, out var uw) &&
                        gw.GgmlType == uw.GgmlType && gw.Ne0 == uw.Ne0)
                    {
                        // ExpertFFN expects a fused gate_up tensor at fusedName. If
                        // MLX view-fusion fails (gate/up not contiguous in GGUF),
                        // fall back to copy — same rationale as FuseGateUpWeights.
                        if (!TryCreateFusedQuantizedWeight(out QuantizedWeight fusedWeight, gw, uw))
                            fusedWeight = QuantizedWeight.ConcatOrCreateCopy(gw, uw);

                        _quantWeights[fusedName] = fusedWeight;
                        _quantWeights.Remove(gateName); gw.Dispose();
                        _quantWeights.Remove(upName); uw.Dispose();
                        fused++;
                    }
                    else if (_weights.TryGetValue(gateName, out var gf) &&
                             _weights.TryGetValue(upName, out var uf))
                    {
                        int gateDim = (int)gf.Sizes[0], upDim = (int)uf.Sizes[0];
                        int inDim = (int)gf.Sizes[1];
                        var fusedTensor = new Tensor(_allocator, DType.Float32, gateDim + upDim, inDim);
                        using (var s0 = fusedTensor.Narrow(0, 0, gateDim)) Ops.Copy(s0, gf);
                        using (var s1 = fusedTensor.Narrow(0, gateDim, upDim)) Ops.Copy(s1, uf);
                        _weights[fusedName] = fusedTensor;
                        _weights.Remove(gateName); gf.Dispose();
                        _weights.Remove(upName); uf.Dispose();
                        fused++;
                    }

                    string gateBias = $"blk.{l}.ffn_gate_exps.{e}.bias";
                    string upBias = $"blk.{l}.ffn_up_exps.{e}.bias";
                    string fusedBias = $"blk.{l}.ffn_gate_up_exps.{e}.bias";
                    if (_weights.TryGetValue(gateBias, out var gb) &&
                        _weights.TryGetValue(upBias, out var ub))
                    {
                        int gbDim = (int)gb.Sizes[1], ubDim = (int)ub.Sizes[1];
                        float[] gbData = TensorToFloatArray(gb);
                        float[] ubData = TensorToFloatArray(ub);
                        float[] fusedData = new float[gbDim + ubDim];
                        Array.Copy(gbData, 0, fusedData, 0, gbDim);
                        Array.Copy(ubData, 0, fusedData, gbDim, ubDim);
                        _weights[fusedBias] = CreateFloatTensor(fusedData, 1, gbDim + ubDim);
                        _weights.Remove(gateBias); gb.Dispose();
                        _weights.Remove(upBias); ub.Dispose();
                    }
                }
            }
            if (fused > 0)
                Console.WriteLine($"  Fused expert Gate+Up projections: {fused}");
        }

        private unsafe void FuseQKVWeights()
        {
            int fused = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                string qName = $"blk.{l}.attn_q.weight";
                string kName = $"blk.{l}.attn_k.weight";
                string vName = $"blk.{l}.attn_v.weight";
                string qkvName = $"blk.{l}.attn_qkv.weight";

                if (_quantWeights.TryGetValue(qName, out var qw) &&
                    _quantWeights.TryGetValue(kName, out var kw) &&
                    _quantWeights.TryGetValue(vName, out var vw) &&
                    qw.GgmlType == kw.GgmlType && kw.GgmlType == vw.GgmlType &&
                    qw.Ne0 == kw.Ne0 && kw.Ne0 == vw.Ne0)
                {
                    if (!TryCreateFusedQuantizedWeight(out QuantizedWeight fusedWeight, qw, kw, vw))
                        continue;

                    _quantWeights[qkvName] = fusedWeight;
                    _quantWeights.Remove(qName); qw.Dispose();
                    _quantWeights.Remove(kName); kw.Dispose();
                    _quantWeights.Remove(vName); vw.Dispose();
                    fused++;
                }
                else if (_weights.TryGetValue(qName, out var qf) &&
                         _weights.TryGetValue(kName, out var kf) &&
                         _weights.TryGetValue(vName, out var vf))
                {
                    int qDim = (int)qf.Sizes[0], kDim = (int)kf.Sizes[0], vDim = (int)vf.Sizes[0];
                    int inDim = (int)qf.Sizes[1];
                    var fusedTensor = new Tensor(_allocator, DType.Float32, qDim + kDim + vDim, inDim);
                    using (var s0 = fusedTensor.Narrow(0, 0, qDim)) Ops.Copy(s0, qf);
                    using (var s1 = fusedTensor.Narrow(0, qDim, kDim)) Ops.Copy(s1, kf);
                    using (var s2 = fusedTensor.Narrow(0, qDim + kDim, vDim)) Ops.Copy(s2, vf);
                    _weights[qkvName] = fusedTensor;
                    _weights.Remove(qName); qf.Dispose();
                    _weights.Remove(kName); kf.Dispose();
                    _weights.Remove(vName); vf.Dispose();
                    fused++;
                }

                string qBias = $"blk.{l}.attn_q.bias";
                string kBias = $"blk.{l}.attn_k.bias";
                string vBias = $"blk.{l}.attn_v.bias";
                string qkvBias = $"blk.{l}.attn_qkv.bias";
                if (_weights.TryGetValue(qBias, out var qb) &&
                    _weights.TryGetValue(kBias, out var kb) &&
                    _weights.TryGetValue(vBias, out var vb))
                {
                    int qbDim = (int)qb.ElementCount();
                    int kbDim = (int)kb.ElementCount();
                    int vbDim = (int)vb.ElementCount();
                    float[] qbData = TensorToFloatArray(qb);
                    float[] kbData = TensorToFloatArray(kb);
                    float[] vbData = TensorToFloatArray(vb);
                    float[] fusedData = new float[qbDim + kbDim + vbDim];
                    Array.Copy(qbData, 0, fusedData, 0, qbDim);
                    Array.Copy(kbData, 0, fusedData, qbDim, kbDim);
                    Array.Copy(vbData, 0, fusedData, qbDim + kbDim, vbDim);
                    _weights[qkvBias] = CreateFloatTensor(fusedData, 1, qbDim + kbDim + vbDim);
                    _weights.Remove(qBias); qb.Dispose();
                    _weights.Remove(kBias); kb.Dispose();
                    _weights.Remove(vBias); vb.Dispose();
                }
            }
            if (fused > 0)
                Console.WriteLine($"  Fused projections: {fused} QKV");
        }

        private void PrecomputeConstants()
        {
            int numLayers = Config.NumLayers;
            _qDim = Config.NumHeads * Config.HeadDim;
            _kDim = Config.NumKVHeads * Config.HeadDim;

            _isQkvFused = _quantWeights.ContainsKey("blk.0.attn_qkv.weight") ||
                           _weights.ContainsKey("blk.0.attn_qkv.weight");

            _layerNames = new string[numLayers][];
            for (int l = 0; l < numLayers; l++)
            {
                string p = $"blk.{l}.";
                if (_isQkvFused)
                {
                    _layerNames[l] = new[]
                    {
                        p + "attn_norm.weight",           // 0
                        p + "attn_qkv.weight",            // 1
                        p + "attn_qkv.bias",              // 2
                        p + "attn_output.weight",          // 3
                        p + "attn_output.bias",            // 4
                        p + "post_attention_norm.weight",  // 5
                        p + "ffn_gate_inp.weight",         // 6
                        p + "ffn_gate_inp.bias",           // 7
                    };
                }
                else
                {
                    _layerNames[l] = new[]
                    {
                        p + "attn_norm.weight",            // 0
                        p + "attn_q.weight",               // 1
                        p + "attn_q.bias",                 // 2
                        p + "attn_output.weight",          // 3
                        p + "attn_output.bias",            // 4
                        p + "post_attention_norm.weight",  // 5
                        p + "ffn_gate_inp.weight",         // 6
                        p + "ffn_gate_inp.bias",           // 7
                        p + "attn_k.weight",               // 8
                        p + "attn_k.bias",                 // 9
                        p + "attn_v.weight",               // 10
                        p + "attn_v.bias",                 // 11
                    };
                }
            }

            _expertNames = new string[numLayers][][];
            for (int l = 0; l < numLayers; l++)
            {
                _expertNames[l] = new string[_numExperts][];
                string p = $"blk.{l}.";
                for (int e = 0; e < _numExperts; e++)
                {
                    _expertNames[l][e] = new[]
                    {
                        p + $"ffn_gate_up_exps.{e}.weight",  // 0
                        p + $"ffn_gate_up_exps.{e}.bias",    // 1
                        p + $"ffn_down_exps.{e}.weight",     // 2
                        p + $"ffn_down_exps.{e}.bias",       // 3
                    };
                }
            }

            _layerSinks = new float[numLayers][];
            for (int l = 0; l < numLayers; l++)
            {
                string sinksKey = $"blk.{l}.attn_sinks.weight";
                if (_weights.TryGetValue(sinksKey, out var sinksTensor))
                    _layerSinks[l] = TensorToFloatArray(sinksTensor);
            }

            int maxBatchTokens = 4096 * _numExpertsUsed;
            _moeExpertCounts = new int[_numExperts];
            _moeExpertOffsets = new int[_numExperts];
            _moeTokenMap = new int[maxBatchTokens];
            _moeWeightMap = new float[maxBatchTokens];
            _moeTopKScratch = new int[_numExpertsUsed];
        }

        #endregion

        private int _kvCacheCapacity;

        private void InitKVCache(int initialSeqLen, int maxSeqLen)
        {
            _maxContextLength = maxSeqLen;
            _kvCacheCapacity = initialSeqLen;
            int numKVHeads = Config.NumKVHeads;
            int headDim = Config.HeadDim;
            // Pick model-aligned default. For F16-quantised GPT-OSS this gives
            // an F16 KV cache (halves cache memory + bandwidth, byte-identical
            // outputs at 1e-3). The fused prefill kernel and the F16-aware
            // decode loop (AttentionDecodeWithSinksF16 below) handle it
            // natively. The legacy per-op prefill path (used only when
            // seqLen > FusedAttnMaxSeqLen, i.e. ubatches > 256) doesn't yet
            // read F16 cache directly via AddmmBatch, so for that path we'd
            // either need to convert on the fly or keep the cache F32. The
            // CLI always uses ubatches that hit the fused path on every
            // shipping GGUF, so the F16 default is safe for benchmark and
            // chat workloads.
            ApplyModelAlignedKvCacheDefault(_quantWeights);
            DType kvDtype = _kvCacheDtype.ToDType();
            _kvCacheK = new Tensor[Config.NumLayers];
            _kvCacheV = new Tensor[Config.NumLayers];
            for (int l = 0; l < Config.NumLayers; l++)
            {
                _kvCacheK[l] = new Tensor(_allocator, kvDtype, numKVHeads, initialSeqLen, headDim);
                _kvCacheV[l] = new Tensor(_allocator, kvDtype, numKVHeads, initialSeqLen, headDim);
                InitializeCacheTensor(_kvCacheK[l]);
                InitializeCacheTensor(_kvCacheV[l]);
            }
            _cacheSeqLen = 0;
        }

        private void EnsureCacheCapacity(int requiredSeqLen)
        {
            if (requiredSeqLen <= _kvCacheCapacity)
                return;
            if (requiredSeqLen > _maxContextLength)
                throw new InvalidOperationException($"Requested sequence length {requiredSeqLen} exceeds configured max context {_maxContextLength}.");

            int newCapacity = Math.Max(_kvCacheCapacity, 1);
            while (newCapacity < requiredSeqLen)
                newCapacity = Math.Min(_maxContextLength, newCapacity * 2);

            int numKVHeads = Config.NumKVHeads;
            int headDim = Config.HeadDim;
            DType kvDtype = _kvCacheDtype.ToDType();
            for (int l = 0; l < Config.NumLayers; l++)
            {
                var newK = new Tensor(_allocator, kvDtype, numKVHeads, newCapacity, headDim);
                var newV = new Tensor(_allocator, kvDtype, numKVHeads, newCapacity, headDim);
                InitializeCacheTensor(newK);
                InitializeCacheTensor(newV);

                if (_cacheSeqLen > 0)
                {
                    using var srcK = _kvCacheK[l].Narrow(1, 0, _cacheSeqLen);
                    using var dstK = newK.Narrow(1, 0, _cacheSeqLen);
                    Ops.Copy(dstK, srcK);

                    using var srcV = _kvCacheV[l].Narrow(1, 0, _cacheSeqLen);
                    using var dstV = newV.Narrow(1, 0, _cacheSeqLen);
                    Ops.Copy(dstV, srcV);
                }

                _kvCacheK[l].Dispose();
                _kvCacheV[l].Dispose();
                _kvCacheK[l] = newK;
                _kvCacheV[l] = newV;
            }

            _kvCacheCapacity = newCapacity;
            Console.WriteLine($"Expanded GPT-OSS attention cache to {newCapacity} tokens.");
        }

        public override void ResetKVCache()
        {
            for (int l = 0; l < Config.NumLayers; l++)
            {
                ResetCacheTensor(_kvCacheK[l]);
                ResetCacheTensor(_kvCacheV[l]);
            }
            _cacheSeqLen = 0;
            _linearTicks = _attnTicks = _normTicks = _embTicks = _lmHeadTicks = _logitsCopyTicks = 0;
            _forwardCount = 0;
            _forwardSw.Reset();
        }

        public override void TruncateKVCache(int tokenCount)
        {
            base.TruncateKVCache(tokenCount);
            for (int l = 0; l < Config.NumLayers; l++)
            {
                InvalidateTensorDeviceCache(_kvCacheK[l]);
                InvalidateTensorDeviceCache(_kvCacheV[l]);
            }
        }

        public override bool SupportsKVStateSnapshot => _kvCacheK != null && _kvCacheV != null;

        public override string KVStateFingerprint =>
            $"gptoss|arch={Config.Architecture}|L={Config.NumLayers}|H={Config.NumHeads}|KV={Config.NumKVHeads}|D={Config.HeadDim}|dtype={_kvCacheDtype.ToShortString()}";

        public override long ComputeKVBlockByteSize(int tokenCount)
            => KvBlockTransfer.ComputeBlockByteSize(_kvCacheK, _kvCacheV, tokenCount);

        public override bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            if (!SupportsKVStateSnapshot)
                return false;
            return KvBlockTransfer.Extract(
                _allocator, _kvCacheK, _kvCacheV, _cacheSeqLen,
                startToken, tokenCount, destination);
        }

        public override bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            if (!SupportsKVStateSnapshot)
                return false;
            EnsureCacheCapacity(destToken + tokenCount);
            if (!KvBlockTransfer.Inject(
                    _allocator, _kvCacheK, _kvCacheV, _cacheSeqLen,
                    destToken, tokenCount, source))
            {
                return false;
            }
            _cacheSeqLen = destToken + tokenCount;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                InvalidateTensorDeviceCache(_kvCacheK[l]);
                InvalidateTensorDeviceCache(_kvCacheV[l]);
            }
            return true;
        }

        // Chunk size for ForwardRefill: long prompts are processed in this-many-token
        // chunks so the per-layer attention-score allocation stays bounded.
        // Override with TS_PREFILL_CHUNK when tuning.
        private static int ResolvePrefillChunkSize()
        {
            string env = Environment.GetEnvironmentVariable("TS_PREFILL_CHUNK");
            if (!string.IsNullOrEmpty(env) && int.TryParse(env, out int v) && v > 0)
                return v;
            return 2048;
        }

        public override float[] ForwardRefill(int[] tokens)
        {
            if (tokens == null || tokens.Length <= 1)
                return Forward(tokens);

            int chunkSize = ResolvePrefillChunkSize();
            int lastIdx = tokens.Length - 1;

            if (tokens.Length <= chunkSize)
                return Forward(tokens);

            for (int pos = 0; pos < lastIdx; pos += chunkSize)
            {
                int chunkLen = Math.Min(chunkSize, lastIdx - pos);
                var chunk = new int[chunkLen];
                Array.Copy(tokens, pos, chunk, 0, chunkLen);
                PrefillWithoutLogits(chunk);
            }
            return Forward(new[] { tokens[lastIdx] });
        }

        private void PrefillWithoutLogits(int[] tokens)
        {
            if (tokens == null || tokens.Length == 0)
                return;

            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;

            EnsureCacheCapacity(startPos + seqLen);

            long t1 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t1;

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                bool isLastLayer = (layer == Config.NumLayers - 1);
                hidden = TransformerBlock(hidden, layer, seqLen, startPos, isLastLayer);
                if (_backend == BackendType.Mlx && (layer + 1) % MlxEvalEveryNLayers == 0
                    && !isLastLayer && hidden != null)
                {
                    MlxFusedOps.TryAsyncEvaluate(hidden);
                }
            }

            hidden.Dispose();
            _cacheSeqLen += seqLen;
            _forwardSw.Stop();
        }

        public override float[] Forward(int[] tokens)
        {
            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;

            EnsureCacheCapacity(startPos + seqLen);

            long t1 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t1;

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                bool isLastLayer = (layer == Config.NumLayers - 1);
                hidden = TransformerBlock(hidden, layer, seqLen, startPos, isLastLayer);
                if (_backend == BackendType.Mlx && (layer + 1) % MlxEvalEveryNLayers == 0
                    && !isLastLayer && hidden != null)
                {
                    MlxFusedOps.TryAsyncEvaluate(hidden);
                }
            }

            Tensor normed = RMSNormOp(hidden, "output_norm.weight");
            hidden.Dispose();

            Tensor lastHidden;
            if (seqLen > 1)
            {
                using var narrowed = normed.Narrow(0, seqLen - 1, 1);
                lastHidden = Ops.NewContiguous(narrowed);
            }
            else
            {
                lastHidden = normed.CopyRef();
            }
            normed.Dispose();

            long t2 = Stopwatch.GetTimestamp();
            Tensor logitsTensor = LinearForward(lastHidden, "output.weight");
            if (logitsTensor == null)
                logitsTensor = LinearForward(lastHidden, "token_embd.weight");
            _lmHeadTicks += Stopwatch.GetTimestamp() - t2;
            lastHidden.Dispose();

            long t3 = Stopwatch.GetTimestamp();
            _logitsBuffer = TensorToFloatArray(logitsTensor);
            _logitsCopyTicks += Stopwatch.GetTimestamp() - t3;
            logitsTensor.Dispose();

            _cacheSeqLen += seqLen;
            _forwardCount++;
            _forwardSw.Stop();
            return _logitsBuffer;
        }

        private Tensor TransformerBlock(Tensor hidden, int layer, int seqLen, int startPos, bool isLastLayer)
        {
            string[] wn = _layerNames[layer];

            // Prefill fused-layer fast path: collapses RMSNorm + fused QKV (+bias)
            // + RoPE + KV-cache append + masked-softmax-with-sinks + attention
            // + output projection (+bias) + residual add into ONE ggml_cgraph
            // dispatch. Replaces the ~10 separate per-op submissions in the
            // legacy Attention() path (each its own Metal command buffer).
            // Mirrors the reference llama.cpp graph in src/models/openai-moe-iswa.cpp
            // and the existing TSGgml_Gemma4LayerPrefill template.
            //
            // Decode (seqLen == 1) and the non-Metal backends still flow through
            // the original per-op path below.
            // Cap the fused path at seqLen <= 256 for now: above that the
            // per-call backend buffer + Metal residency overhead from N
            // attention layers + N MoE FFN layers in flight exceeds the
            // recommendedMaxWorkingSetSize on Apple Silicon and triggers
            // kIOGPUCommandBufferCallbackErrorOutOfMemory in subsequent kernels.
            // The legacy per-op path is still competitive at long seqLen
            // (each per-op kernel reuses small per-op intermediate buffers
            // via ggml-pool) and remains the default for those. A future
            // wave will rework per-call buffer reuse (e.g. via ggml_gallocr)
            // to lift this cap.
            const int FusedAttnMaxSeqLen = 256;
            bool fusedAttnApplied = false;
            if (seqLen > 1 && seqLen <= FusedAttnMaxSeqLen && IsGgmlBackend
                && TryFusedAttnLayerPrefill(hidden, layer, wn, seqLen, startPos))
            {
                fusedAttnApplied = true;
            }

            if (!fusedAttnApplied)
            {
                Tensor normed = RMSNormOp(hidden, wn[0]);
                Tensor attnOut = Attention(normed, layer, wn, seqLen, startPos);
                normed.Dispose();
                Ops.Add(hidden, hidden, attnOut);
                attnOut.Dispose();
            }

            int moeSeqLen = seqLen;
            Tensor moeInput = hidden;
            if (isLastLayer && seqLen > 1)
            {
                using var lastRow = hidden.Narrow(0, seqLen - 1, 1);
                moeInput = Ops.NewContiguous(lastRow);
                moeSeqLen = 1;
            }

            Tensor normed2 = RMSNormOp(moeInput, wn[5]);
            Tensor moeOut = MoEForward(normed2, layer, moeSeqLen);
            normed2.Dispose();

            if (isLastLayer && seqLen > 1)
            {
                unsafe
                {
                    float* hidPtr = GetFloatPtr(hidden);
                    float* moePtr = GetFloatPtr(moeOut);
                    int dim = Config.HiddenSize;
                    long offset = (long)(seqLen - 1) * dim;
                    for (int d = 0; d < dim; d++)
                        hidPtr[offset + d] += moePtr[d];
                }
                moeInput.Dispose();
            }
            else
            {
                Ops.Add(hidden, hidden, moeOut);
            }
            moeOut.Dispose();

            return hidden;
        }

        #region Attention

        /// <summary>
        /// Fused per-layer prefill kernel (TSGgml_GptOssAttentionLayerPrefill).
        ///
        /// Runs the full attention block (input RMSNorm + fused QKV (+bias) +
        /// RoPE + KV-cache append + causal/SWA mask + softmax-with-sinks +
        /// attention + output projection (+bias) + residual add) as ONE
        /// ggml_cgraph dispatch per layer, writing the residual back into the
        /// caller's `hidden` buffer in place. Returns true on success.
        ///
        /// Returns false (and does NOT touch `hidden`) when:
        ///  - any of the required weights / norm tensors aren't loaded for
        ///    this layer, or
        ///  - the QKV weight isn't a quantized weight (the kernel currently
        ///    requires the quantized-weight CacheKey for zero-copy binding;
        ///    falling back to the F32 weight path is supported by the C# code
        ///    below so we just refuse the fused path here).
        ///
        /// Caller is expected to fall back to the legacy per-op path.
        /// </summary>
        private unsafe bool TryFusedAttnLayerPrefill(
            Tensor hidden, int layer, string[] wn, int seqLen, int startPos)
        {
            // The kernel binds quantized weights via QuantizedWeight.CacheKey
            // (which becomes a stable pointer that the cacheable-buffer path can
            // recognise across calls). For F32 weights we'd need to extend the
            // kernel; for now we only enable the fast path when the QKV / O
            // weights are quantized, which covers every Q*_0 / Q*_K / IQ* GGUF
            // we ship benchmarks for.
            if (!_quantWeights.TryGetValue(wn[1], out var qkvQw)) return false;
            if (!_quantWeights.TryGetValue(wn[3], out var oQw)) return false;
            if (!_weights.TryGetValue(wn[0], out var attnNormW)) return false;

            // attn_qkv.bias / attn_output.bias are optional in the GGUF schema
            // but present on every shipping GPT-OSS model (the bias arrays
            // were already split per-projection earlier in the loader).
            _weights.TryGetValue(wn[2], out var qkvBias);
            _weights.TryGetValue(wn[4], out var oBias);

            // Sliding-window for even layers; full causal for odd layers.
            // Mirrors the legacy Attention() path's `bool isSWA = (layer % 2 == 0)`.
            bool isSwa = (layer % 2 == 0);
            float[] sinks = _layerSinks?[layer];

            // For the separate-Q/K/V path we'd need the K/V quantized weights too.
            // The kernel signature already accepts them; we wire up here.
            QuantizedWeight kQw = null, vQw = null;
            Tensor kBias = null, vBias = null;
            if (!_isQkvFused)
            {
                if (!_quantWeights.TryGetValue(wn[8], out kQw)) return false;
                if (!_quantWeights.TryGetValue(wn[10], out vQw)) return false;
                _weights.TryGetValue(wn[9], out kBias);
                _weights.TryGetValue(wn[11], out vBias);
            }

            // KV cache size and cache-dtype enum.
            int cacheSize = (int)_kvCacheK[layer].Sizes[1];
            int kvCacheTypeId = _kvCacheDtype.GgmlType();

            // The kernel writes to the F32 KV cache via ggml_cpy(F32->F32) and
            // to F16 via ggml_cpy(F32->F16). Quantized cache types aren't yet
            // supported here, so fall back when we'd otherwise wedge.
            if (kvCacheTypeId != 0 /* F32 */ && kvCacheTypeId != 1 /* F16 */)
                return false;

            try
            {
                long t0 = Stopwatch.GetTimestamp();
                GgmlBasicOps.GptOssAttentionLayerPrefill(
                    (IntPtr)GetFloatPtr(hidden),
                    Config.HiddenSize, seqLen,
                    (IntPtr)GetFloatPtr(attnNormW),
                    qkvQw.CacheKey, qkvQw.GgmlType, qkvQw.Ne0, qkvQw.Ne1, qkvQw.RawBytes,
                    qkvBias != null ? (IntPtr)GetFloatPtr(qkvBias) : IntPtr.Zero,
                    _isQkvFused ? 1 : 0,
                    kQw != null ? kQw.CacheKey : IntPtr.Zero,
                    kQw?.GgmlType ?? 0, kQw?.Ne0 ?? 0, kQw?.Ne1 ?? 0, kQw?.RawBytes ?? 0,
                    kBias != null ? (IntPtr)GetFloatPtr(kBias) : IntPtr.Zero,
                    vQw != null ? vQw.CacheKey : IntPtr.Zero,
                    vQw?.GgmlType ?? 0, vQw?.Ne0 ?? 0, vQw?.Ne1 ?? 0, vQw?.RawBytes ?? 0,
                    vBias != null ? (IntPtr)GetFloatPtr(vBias) : IntPtr.Zero,
                    oQw.CacheKey, oQw.GgmlType, oQw.Ne0, oQw.Ne1, oQw.RawBytes,
                    oBias != null ? (IntPtr)GetFloatPtr(oBias) : IntPtr.Zero,
                    TensorComputePrimitives.GetStoragePointer(_kvCacheK[layer]),
                    TensorComputePrimitives.GetStoragePointer(_kvCacheV[layer]),
                    Config.NumHeads, Config.NumKVHeads, Config.HeadDim,
                    cacheSize, startPos,
                    isSwa ? 1 : 0, _slidingWindow,
                    sinks != null ? (IntPtr)GetFloatArrayPtr(sinks, layer) : IntPtr.Zero,
                    Config.RopeBase, 1.0f / Config.RopeScale, Config.HeadDim,
                    Config.OriginalContextLength,
                    kvCacheTypeId, Config.Eps);
                _attnTicks += Stopwatch.GetTimestamp() - t0;
                InvalidateTensorDeviceCache(hidden);
                return true;
            }
            catch
            {
                return false;
            }
        }

        // Returns an MLX-backed [numHeads] Float32 tensor populated from
        // the host-side `_layerSinks[layer]` array, allocated on first
        // call per layer and reused thereafter.
        private Tensor GetOrCreateSinksMlxTensor(int layer, float[] sinksArray, int numHeads)
        {
            if (_layerSinksMlx == null)
                _layerSinksMlx = new Tensor[Config.NumLayers];
            if (_layerSinksMlx[layer] != null)
                return _layerSinksMlx[layer];
            if (sinksArray == null || sinksArray.Length < numHeads)
                return null;

            var t = new Tensor(_allocator, DType.Float32, numHeads);
            t.SetElementsAsFloat(sinksArray);
            _layerSinksMlx[layer] = t;
            return t;
        }

        // Per-layer cached pinned-handle for sinks arrays so we can hand the
        // kernel a stable IntPtr that the cacheable-host-ptr path recognises
        // across calls (and pin only once per layer).
        private System.Runtime.InteropServices.GCHandle[] _sinksHandles;

        private unsafe float* GetFloatArrayPtr(float[] arr, int layer)
        {
            if (arr == null) return null;
            if (_sinksHandles == null)
                _sinksHandles = new System.Runtime.InteropServices.GCHandle[Config.NumLayers];
            if (!_sinksHandles[layer].IsAllocated)
            {
                _sinksHandles[layer] = System.Runtime.InteropServices.GCHandle.Alloc(arr, System.Runtime.InteropServices.GCHandleType.Pinned);
            }
            return (float*)_sinksHandles[layer].AddrOfPinnedObject();
        }

        private Tensor Attention(Tensor input, int layer, string[] wn, int seqLen, int startPos)
        {
            int numHeads = Config.NumHeads;
            int numKVHeads = Config.NumKVHeads;
            int headDim = Config.HeadDim;
            int totalSeqLen = startPos + seqLen;
            float scale = 1.0f / MathF.Sqrt(headDim);
            bool isSWA = (layer % 2 == 0);

            Tensor qTensor, kTensor, vTensor;

            if (_isQkvFused)
            {
                Tensor qkvFused = LinearForwardWithBias(input, wn[1], wn[2]);

                if (seqLen == 1)
                {
                    qTensor = qkvFused.Narrow(1, 0, _qDim);
                    kTensor = qkvFused.Narrow(1, _qDim, _kDim);
                    vTensor = qkvFused.Narrow(1, _qDim + _kDim, _kDim);
                    qkvFused.Dispose();
                }
                else
                {
                    qTensor = SliceColumnsContiguous(qkvFused, 0, _qDim);
                    kTensor = SliceColumnsContiguous(qkvFused, _qDim, _kDim);
                    vTensor = SliceColumnsContiguous(qkvFused, _qDim + _kDim, _kDim);
                    qkvFused.Dispose();
                }
            }
            else
            {
                qTensor = LinearForwardWithBias(input, wn[1], wn[2]);
                kTensor = LinearForwardWithBias(input, wn[8], wn[9]);
                vTensor = LinearForwardWithBias(input, wn[10], wn[11]);
            }

            qTensor = ApplyRoPEInPlace(qTensor, numHeads, headDim, seqLen, startPos);
            kTensor = ApplyRoPEInPlace(kTensor, numKVHeads, headDim, seqLen, startPos);

            float[] sinks = _layerSinks[layer];

            long t0 = Stopwatch.GetTimestamp();

            if (seqLen == 1)
            {
                CopyToCacheDecode(_kvCacheK[layer], kTensor, _kvCacheV[layer], vTensor,
                    numKVHeads, headDim, startPos);
                kTensor.Dispose();
                vTensor.Dispose();

                var attnResult = new Tensor(_allocator, DType.Float32, 1, numHeads * headDim);

                // MLX path: keep K/V on device, run the sinks-aware decode
                // attention via a custom Metal kernel. Avoids the per-layer
                // device→host KV cache pull that AttentionDecodeWithSinks
                // triggers via GetFloatPtr/GetHalfPointer. Only worth it
                // for long context — the kernel's per-K-step barriers
                // outweigh the cache download cost for short kvLen, where
                // the host SIMD CPU path is faster. Threshold tunable via
                // TS_MLX_SINKS_ATTN_MIN_KV_LEN (default 2048).
                bool attnOk = false;
                if (_backend == BackendType.Cuda)
                {
                    Tensor sinksCuda = sinks != null ? GetOrCreateSinksMlxTensor(layer, sinks, numHeads) : null;
                    int attendStart = isSWA ? Math.Max(0, totalSeqLen - _slidingWindow) : 0;
                    int attendLen = totalSeqLen - attendStart;
                    attnOk = CudaFusedOps.TryGqaDecodeAttentionWithSinks(
                        attnResult, qTensor,
                        _kvCacheK[layer], _kvCacheV[layer], sinksCuda,
                        numHeads, numKVHeads, headDim,
                        attendStart, attendLen, _kvCacheCapacity,
                        circular: false, scale);
                }
                if (_backend == BackendType.Mlx
                    && sinks != null
                    && totalSeqLen >= MlxSinksAttnMinKvLen)
                {
                    Tensor sinksMlx = GetOrCreateSinksMlxTensor(layer, sinks, numHeads);
                    if (sinksMlx != null)
                    {
                        int sw = isSWA ? _slidingWindow : 0;
                        attnOk = MlxFusedOps.TryDecodeAttentionWithSinks(
                            attnResult, qTensor,
                            _kvCacheK[layer], _kvCacheV[layer], sinksMlx,
                            numHeads, numKVHeads, headDim,
                            _kvCacheCapacity, totalSeqLen, sw, scale);
                    }
                }
                if (!attnOk)
                {
                    AttentionDecodeWithSinks(qTensor, _kvCacheK[layer], _kvCacheV[layer],
                        attnResult, numHeads, numKVHeads, headDim, totalSeqLen, scale, sinks, isSWA);
                }
                qTensor.Dispose();

                _attnTicks += Stopwatch.GetTimestamp() - t0;

                Tensor decodeOut = LinearForwardWithBias(attnResult, wn[3], wn[4]);
                attnResult.Dispose();
                return decodeOut;
            }

            // Prefill path
            Tensor qHeads = ReshapeToHeads(qTensor, numHeads, seqLen, headDim);
            qTensor.Dispose();
            Tensor kHeads = ReshapeToHeads(kTensor, numKVHeads, seqLen, headDim);
            kTensor.Dispose();
            Tensor vHeads = ReshapeToHeads(vTensor, numKVHeads, seqLen, headDim);
            vTensor.Dispose();

            CopyToCache(_kvCacheK[layer], kHeads, startPos, seqLen);
            CopyToCache(_kvCacheV[layer], vHeads, startPos, seqLen);
            kHeads.Dispose();
            vHeads.Dispose();

            if (_backend == BackendType.Cuda)
            {
                var fusedAttention = new Tensor(_allocator, DType.Float32, seqLen, numHeads * headDim);
                Tensor sinksCuda = sinks != null ? GetOrCreateSinksMlxTensor(layer, sinks, numHeads) : null;
                if (CudaFusedOps.TryGqaPrefillAttentionWithSinks(
                    fusedAttention,
                    qHeads,
                    _kvCacheK[layer],
                    _kvCacheV[layer],
                    sinksCuda,
                    numHeads,
                    numKVHeads,
                    headDim,
                    seqLen,
                    totalSeqLen,
                    _kvCacheCapacity,
                    startPos,
                    isSWA ? _slidingWindow : 0,
                    scale))
                {
                    qHeads.Dispose();
                    _attnTicks += Stopwatch.GetTimestamp() - t0;

                    Tensor fusedOutput = LinearForwardWithBias(fusedAttention, wn[3], wn[4]);
                    fusedAttention.Dispose();
                    return fusedOutput;
                }
                fusedAttention.Dispose();
            }

            int groupSize = numHeads / numKVHeads;
            Tensor kExpanded = ExpandKVHeads(_kvCacheK[layer], groupSize, totalSeqLen);
            Tensor vExpanded = ExpandKVHeads(_kvCacheV[layer], groupSize, totalSeqLen);

            using var kT = kExpanded.Transpose(1, 2);
            var scores = new Tensor(_allocator, DType.Float32, numHeads, seqLen, totalSeqLen);
            Ops.AddmmBatch(scores, 0, scores, scale, qHeads, kT);
            qHeads.Dispose();
            kExpanded.Dispose();

            // Fused causal+SWA mask + softmax + attention sinks on GPU. Replaces the
            // GPU AddCausalMask + the two CPU loops (ApplySWAMask /
            // ApplySoftmaxWithSinks) which together dominated GptOss prefill —
            // ~76% of total time on pp2048, mostly from the ~6-billion-element
            // single-threaded MathF.Exp loop in ApplySoftmaxWithSinks.
            //
            // Scores are already pre-scaled by 1/sqrt(headDim) in AddmmBatch above,
            // so we pass scale=1.0 here.
            if (IsGgmlBackend)
            {
                GgmlBasicOps.AttentionSoftmaxWithSinks(
                    scores,
                    sinks,
                    numHeads: numHeads,
                    seqLen: seqLen,
                    kvLen: totalSeqLen,
                    maskStartPos: startPos,
                    slidingWindow: isSWA ? _slidingWindow : 0,
                    scale: 1.0f);
            }
            else if (_backend == BackendType.Cuda &&
                     CudaFusedOps.TryAttentionSoftmaxWithSinks(
                         scores,
                         sinks != null ? GetOrCreateSinksMlxTensor(layer, sinks, numHeads) : null,
                         numHeads,
                         seqLen,
                         totalSeqLen,
                         startPos,
                         isSWA ? _slidingWindow : 0,
                         1.0f))
            {
            }
            else
            {
                Ops.AddCausalMask(scores, seqLen, startPos, float.NegativeInfinity);
                if (isSWA)
                    ApplySWAMask(scores, numHeads, seqLen, totalSeqLen, startPos);
                ApplySoftmaxWithSinks(scores, numHeads, seqLen, totalSeqLen, sinks);
            }

            var attnOut = new Tensor(_allocator, DType.Float32, numHeads, seqLen, headDim);
            Ops.AddmmBatch(attnOut, 0, attnOut, 1.0f, scores, vExpanded);
            scores.Dispose();
            vExpanded.Dispose();

            Tensor flatOutput = ReshapeFromHeads(attnOut, numHeads, seqLen, headDim);
            attnOut.Dispose();

            _attnTicks += Stopwatch.GetTimestamp() - t0;

            Tensor output = LinearForwardWithBias(flatOutput, wn[3], wn[4]);
            flatOutput.Dispose();
            return output;
        }

        private Tensor SliceColumnsContiguous(Tensor src, int colOffset, int width)
        {
            var result = new Tensor(_allocator, DType.Float32, src.Sizes[0], width);
            if (CudaFusedOps.TrySliceColumns(result, src, colOffset, width))
                return result;
            result.Dispose();

            using var view = src.Narrow(1, colOffset, width);
            return Ops.NewContiguous(view);
        }

        private unsafe void ApplySWAMask(Tensor scores, int numHeads, int seqLen, int totalSeqLen, int startPos)
        {
            float* ptr = GetFloatPtr(scores);
            for (int h = 0; h < numHeads; h++)
            {
                for (int q = 0; q < seqLen; q++)
                {
                    int qPos = startPos + q;
                    float* row = ptr + ((long)h * seqLen + q) * totalSeqLen;
                    int limit = qPos - _slidingWindow + 1;
                    for (int k = 0; k < totalSeqLen && k < limit; k++)
                        row[k] = float.NegativeInfinity;
                }
            }
        }

        private unsafe void ApplySoftmaxWithSinks(Tensor scores, int numHeads, int seqLen, int totalSeqLen, float[] sinks)
        {
            if (sinks == null)
            {
                Ops.Softmax(scores, scores);
                return;
            }

            float* ptr = GetFloatPtr(scores);
            for (int h = 0; h < numHeads; h++)
            {
                float sinkVal = sinks[h];
                for (int s = 0; s < seqLen; s++)
                {
                    float* row = ptr + ((long)h * seqLen + s) * totalSeqLen;

                    float maxVal = sinkVal;
                    for (int t = 0; t < totalSeqLen; t++)
                        if (row[t] > maxVal) maxVal = row[t];

                    float sumExp = MathF.Exp(sinkVal - maxVal);
                    for (int t = 0; t < totalSeqLen; t++)
                    {
                        row[t] = MathF.Exp(row[t] - maxVal);
                        sumExp += row[t];
                    }

                    float invSum = 1.0f / sumExp;
                    for (int t = 0; t < totalSeqLen; t++)
                        row[t] *= invSum;
                }
            }
        }

        private unsafe void AttentionDecodeWithSinks(Tensor q, Tensor kCache, Tensor vCache,
            Tensor result, int numHeads, int numKVHeads, int headDim, int totalSeqLen, float scale, float[] sinks, bool isSWA)
        {
            // Dispatch on KV cache dtype. The fast paths (F32 below, F16 just
            // beneath) are CPU implementations of single-token GQA attention
            // with attention sinks (per-head learned bias added as a virtual
            // token in the softmax max/exp-sum) and optional sliding-window
            // attention (SWA: keys older than `slidingWindow` positions are
            // masked out). Used as a fallback when the GGML decode kernel is
            // unavailable for this layer (e.g. F16 cache + sinks not yet
            // covered by TSGgml_FlashAttnDecodeF32 sink-adapted variant).
            if (kCache.ElementType == DType.Float16 && vCache.ElementType == DType.Float16)
            {
                AttentionDecodeWithSinksF16(q, kCache, vCache, result,
                    numHeads, numKVHeads, headDim, totalSeqLen, scale, sinks, isSWA);
                return;
            }

            float* qPtr = GetFloatPtr(q);
            float* kPtr = GetFloatPtr(kCache);
            float* vPtr = GetFloatPtr(vCache);
            float* rPtr = GetFloatPtr(result);
            int maxSeqLen = (int)kCache.Sizes[1];
            int groupSize = numHeads / numKVHeads;

            int startT = isSWA ? Math.Max(0, totalSeqLen - _slidingWindow) : 0;
            int numScores = totalSeqLen - startT;
            float* scores = stackalloc float[numScores];

            for (int h = 0; h < numHeads; h++)
            {
                float* qHead = qPtr + h * headDim;
                int kvHead = h / groupSize;
                float* kHead = kPtr + kvHead * maxSeqLen * headDim;
                float* vHead = vPtr + kvHead * maxSeqLen * headDim;

                float maxScore = (sinks != null) ? sinks[h] : float.NegativeInfinity;
                for (int i = 0; i < numScores; i++)
                {
                    int t = startT + i;
                    float s = VecDot(qHead, kHead + t * headDim, headDim) * scale;
                    scores[i] = s;
                    if (s > maxScore) maxScore = s;
                }

                float sumExp = (sinks != null) ? MathF.Exp(sinks[h] - maxScore) : 0f;
                for (int i = 0; i < numScores; i++)
                {
                    float e = MathF.Exp(scores[i] - maxScore);
                    scores[i] = e;
                    sumExp += e;
                }
                float invSum = 1.0f / sumExp;
                for (int i = 0; i < numScores; i++)
                    scores[i] *= invSum;

                float* rHead = rPtr + h * headDim;
                VecZero(rHead, headDim);
                for (int i = 0; i < numScores; i++)
                    VecScaleAdd(rHead, vHead + (startT + i) * headDim, scores[i], headDim);
            }
        }

        /// <summary>
        /// F16-cache variant of <see cref="AttentionDecodeWithSinks"/>. Reads
        /// K/V values as ushort, converts to F32 inside the dot/scale-add hot
        /// loops via <see cref="TensorComputePrimitives"/>. Identical math to
        /// the F32 path; only the cache load is widened. Parallelised over
        /// query heads to amortise the per-head F16-&gt;F32 widening.
        /// </summary>
        private unsafe void AttentionDecodeWithSinksF16(Tensor q, Tensor kCache, Tensor vCache,
            Tensor result, int numHeads, int numKVHeads, int headDim, int totalSeqLen, float scale, float[] sinks, bool isSWA)
        {
            long qPtrL = (long)GetFloatPtr(q);
            long kPtrL = (long)TensorComputePrimitives.GetHalfPointer(kCache);
            long vPtrL = (long)TensorComputePrimitives.GetHalfPointer(vCache);
            long rPtrL = (long)GetFloatPtr(result);
            int maxSeqLen = (int)kCache.Sizes[1];
            int groupSize = numHeads / numKVHeads;

            int startT = isSWA ? Math.Max(0, totalSeqLen - _slidingWindow) : 0;
            int numScores = totalSeqLen - startT;
            int headDimLocal = headDim;
            int maxSeqLenLocal = maxSeqLen;
            int groupSizeLocal = groupSize;
            int startTLocal = startT;
            int numScoresLocal = numScores;
            float scaleLocal = scale;
            float[] sinksLocal = sinks;

            Parallel.For(0, numHeads, h =>
            {
                float* qPtr = (float*)qPtrL;
                ushort* kPtr = (ushort*)kPtrL;
                ushort* vPtr = (ushort*)vPtrL;
                float* rPtr = (float*)rPtrL;

                float* qHead = qPtr + h * headDimLocal;
                int kvHead = h / groupSizeLocal;
                ushort* kHead = kPtr + kvHead * maxSeqLenLocal * headDimLocal;
                ushort* vHead = vPtr + kvHead * maxSeqLenLocal * headDimLocal;

                float* scores = stackalloc float[numScoresLocal];
                float* vF32 = stackalloc float[headDimLocal];

                float maxScore = (sinksLocal != null) ? sinksLocal[h] : float.NegativeInfinity;
                for (int i = 0; i < numScoresLocal; i++)
                {
                    int t = startTLocal + i;
                    float s = TensorComputePrimitives.DotF32F16(qHead, kHead + t * headDimLocal, headDimLocal) * scaleLocal;
                    scores[i] = s;
                    if (s > maxScore) maxScore = s;
                }

                float sumExp = (sinksLocal != null) ? MathF.Exp(sinksLocal[h] - maxScore) : 0f;
                for (int i = 0; i < numScoresLocal; i++)
                {
                    float e = MathF.Exp(scores[i] - maxScore);
                    scores[i] = e;
                    sumExp += e;
                }
                float invSum = 1.0f / sumExp;
                for (int i = 0; i < numScoresLocal; i++)
                    scores[i] *= invSum;

                float* rHead = rPtr + h * headDimLocal;
                VecZero(rHead, headDimLocal);
                for (int i = 0; i < numScoresLocal; i++)
                {
                    TensorComputePrimitives.F16ToF32(vF32, vHead + (startTLocal + i) * headDimLocal, headDimLocal);
                    VecScaleAdd(rHead, vF32, scores[i], headDimLocal);
                }
            });
        }

        private Tensor ApplyRoPEInPlace(Tensor data, int numHeads, int headDim, int seqLen, int startPos)
        {
            int totalRows = seqLen * numHeads;
            int[] positions = new int[totalRows];
            for (int s = 0; s < seqLen; s++)
                for (int h = 0; h < numHeads; h++)
                    positions[s * numHeads + h] = startPos + s;
            using var posTensor = CreateIntTensor(positions, totalRows);

            using var reshaped = data.View(1, seqLen, numHeads, headDim);
            Tensor result = Ops.RoPEEx(
                null, reshaped, posTensor, headDim, 2,
                Config.OriginalContextLength,
                Config.RopeBase, 1.0f / Config.RopeScale,
                1.0f, 1.0f, 32.0f, 1.0f);

            data.Dispose();
            Tensor flat = result.View(seqLen, numHeads * headDim);
            result.Dispose();
            return flat;
        }

        #endregion

        #region MoE

        private unsafe Tensor MoEForward(Tensor hiddenState, int layer, int seqLen)
        {
            string[] wn = _layerNames[layer];
            var (routingWeights, selectedExperts) = MoERoute(hiddenState, wn[6], wn[7], seqLen);

            int hiddenDim = (int)hiddenState.Sizes[1];
            var output = new Tensor(_allocator, DType.Float32, seqLen, hiddenDim);
            Ops.Fill(output, 0f);

            if (seqLen == 1)
            {
                MoEForwardSingleToken(hiddenState, output, routingWeights, selectedExperts, layer, hiddenDim);
                return output;
            }

            MoEForwardBatched(hiddenState, output, routingWeights, selectedExperts, layer, seqLen, hiddenDim);
            return output;
        }

        private unsafe void MoEForwardSingleToken(Tensor hiddenState, Tensor output,
            float[] routingWeights, int[] selectedExperts, int layer, int hiddenDim)
        {
            // Decode fast path: route the single-token MoE FFN through the
            // ggml_mul_mat_id-based fused kernel (TSGgml_MoEFFNPrefillSwiGLU)
            // when the stacked weights are loaded. This collapses the
            // per-active-expert loop (4 expert × 3 ops = ~12 graph dispatches
            // per layer per token) into ONE dispatch using
            // ggml_mul_mat_id with a [1, n_used] ids tensor, mirroring
            // llama.cpp's `build_moe_ffn` and matching the prefill path.
            // Falls back to the per-expert CPU loop only when the stacked
            // weights aren't available for this layer.
            if (_layerStackedReady != 0
                && IsGgmlBackend
                && _layerStackedGate != null && _layerStackedGate[layer] != null
                && _layerStackedUp != null && _layerStackedUp[layer] != null
                && _layerStackedDown != null && _layerStackedDown[layer] != null)
            {
                if (TryMoEPrefillFused(hiddenState, output, routingWeights, selectedExperts, layer, /*seqLen=*/1, hiddenDim))
                    return;
            }

            float* outputPtr = _backend == BackendType.Cuda ? null : GetFloatPtr(output);

            for (int e = 0; e < _numExpertsUsed; e++)
            {
                int expertIdx = selectedExperts[e];
                float weight = routingWeights[e];
                string[] en = _expertNames[layer][expertIdx];

                Tensor expertOut = ExpertFFN(hiddenState, en[0], en[1], en[2], en[3], 1);
                if (_backend == BackendType.Cuda)
                    Ops.AddMulV(output, output, expertOut, weight);
                else
                {
                    float* expertPtr = GetFloatPtr(expertOut);
                    VecScaleAdd(outputPtr, expertPtr, weight, hiddenDim);
                }
                expertOut.Dispose();
            }
        }

        private unsafe void MoEForwardBatched(Tensor hiddenState, Tensor output,
            float[] routingWeights, int[] selectedExperts, int layer, int seqLen, int hiddenDim)
        {
            // Fast path: collapse the entire MoE FFN body into a single ggml_cgraph
            // built from ggml_mul_mat_id + ggml_add_id + swiglu_oai. Replaces the
            // per-expert loop below (one ExpertFFN call per active expert) with
            // a single dispatch per layer. Mirrors llama.cpp's `build_moe_ffn`
            // and is required to close the prefill gap on MoE models like GPT-OSS.
            //
            // Skip for very long prefills: with only 32 experts the legacy
            // batched-by-expert path keeps each per-expert matmul fat (count >> 1)
            // so the per-call ggml graph build / Metal command-buffer overhead
            // of the fused path is no longer a win, and on GPT-OSS specifically
            // it perturbs the Metal scheduler enough to slow down the
            // immediately-following SWA / full attention layers. The crossover
            // is around seq_len ≈ 1024 on M-series Metal; below that the
            // fused path is consistently faster.
            const int FusedMoEMaxSeqLen = 1024;
            if (seqLen <= FusedMoEMaxSeqLen
                && _layerStackedReady != 0
                && IsGgmlBackend
                && _layerStackedGate != null && _layerStackedGate[layer] != null
                && _layerStackedUp != null && _layerStackedUp[layer] != null
                && _layerStackedDown != null && _layerStackedDown[layer] != null)
            {
                if (TryMoEPrefillFused(hiddenState, output, routingWeights, selectedExperts, layer, seqLen, hiddenDim))
                    return;
            }

            float* inputPtr = GetFloatPtr(hiddenState);
            float* outputPtr = GetFloatPtr(output);

            int totalAssignments = seqLen * _numExpertsUsed;
            int[] expertCounts = _moeExpertCounts;
            int[] expertOffsets = _moeExpertOffsets;
            int[] tokenMap = _moeTokenMap;
            float[] weightMap = _moeWeightMap;

            if (totalAssignments > tokenMap.Length)
            {
                tokenMap = _moeTokenMap = new int[totalAssignments];
                weightMap = _moeWeightMap = new float[totalAssignments];
            }

            Array.Clear(expertCounts, 0, _numExperts);

            for (int s = 0; s < seqLen; s++)
                for (int k = 0; k < _numExpertsUsed; k++)
                    expertCounts[selectedExperts[s * _numExpertsUsed + k]]++;

            expertOffsets[0] = 0;
            for (int e = 1; e < _numExperts; e++)
                expertOffsets[e] = expertOffsets[e - 1] + expertCounts[e - 1];

            int[] fillPos = _moeExpertCounts;
            Array.Copy(expertOffsets, fillPos, _numExperts);

            for (int s = 0; s < seqLen; s++)
            {
                for (int k = 0; k < _numExpertsUsed; k++)
                {
                    int expertIdx = selectedExperts[s * _numExpertsUsed + k];
                    int pos = fillPos[expertIdx]++;
                    tokenMap[pos] = s;
                    weightMap[pos] = routingWeights[s * _numExpertsUsed + k];
                }
            }

            for (int e = 0; e < _numExperts; e++)
            {
                int count = (e < _numExperts - 1) ? expertOffsets[e + 1] - expertOffsets[e]
                                                   : totalAssignments - expertOffsets[e];
                if (count == 0) continue;

                int offset = expertOffsets[e];
                string[] en = _expertNames[layer][e];

                var batchInput = new Tensor(_allocator, DType.Float32, count, hiddenDim);
                float* batchPtr = GetFloatPtr(batchInput);

                long rowBytes = hiddenDim * sizeof(float);
                for (int i = 0; i < count; i++)
                {
                    int tokenIdx = tokenMap[offset + i];
                    Buffer.MemoryCopy(inputPtr + (long)tokenIdx * hiddenDim,
                        batchPtr + (long)i * hiddenDim, rowBytes, rowBytes);
                }

                Tensor expertOut = ExpertFFN(batchInput, en[0], en[1], en[2], en[3], count);
                batchInput.Dispose();

                float* expertOutPtr = GetFloatPtr(expertOut);
                for (int i = 0; i < count; i++)
                {
                    int tokenIdx = tokenMap[offset + i];
                    float weight = weightMap[offset + i];
                    VecScaleAdd(outputPtr + (long)tokenIdx * hiddenDim,
                        expertOutPtr + (long)i * hiddenDim, weight, hiddenDim);
                }
                expertOut.Dispose();
            }
        }

        /// <summary>
        /// Fused MoE prefill via the GgmlBasicOps.MoEFFNPrefillSwiGLU kernel.
        /// Replaces the per-active-expert ExpertFFN loop (~num_experts × ~3
        /// graph dispatches per layer) with a single graph dispatch that
        /// performs gate + up + add_id(bias) + swiglu_oai + down + add_id(bias)
        /// + expert weighting + aggregation using ggml_mul_mat_id +
        /// ggml_add_id, mirroring llama.cpp's build_moe_ffn for clamped-SiLU.
        ///
        /// Returns true on success (output has been written; routingWeights
        /// scaling is applied by the kernel). Returns false when the kernel
        /// can't handle the layout and the caller should fall back to the
        /// legacy batched-by-expert path.
        /// </summary>
        private unsafe bool TryMoEPrefillFused(
            Tensor hiddenState,
            Tensor output,
            float[] routingWeights,
            int[] selectedExperts,
            int layer,
            int seqLen,
            int hiddenDim)
        {
            var gateW = _layerStackedGate[layer];
            var upW = _layerStackedUp[layer];
            var downW = _layerStackedDown[layer];

            float[] gateBias = null;
            float[] upBias = null;
            if (_layerGateUpBiasStacked != null && _layerGateUpBiasStacked[layer] != null)
            {
                // The gate-up bias was stacked as [(2*nFf), num_experts]
                // gate-then-up. Split into separate gate and up arrays so the
                // kernel can use the SEPARATE gate/up weight path (which lets
                // us reuse the original 3D `_exps.weight` blocks zero-copy).
                int nFf = _expertFfnLength;
                float[] fused = _layerGateUpBiasStacked[layer];
                gateBias = new float[nFf * _numExperts];
                upBias = new float[nFf * _numExperts];
                for (int e = 0; e < _numExperts; e++)
                {
                    Array.Copy(fused, e * 2 * nFf, gateBias, e * nFf, nFf);
                    Array.Copy(fused, e * 2 * nFf + nFf, upBias, e * nFf, nFf);
                }
            }
            float[] downBias = (_layerDownBiasStacked != null) ? _layerDownBiasStacked[layer] : null;

            try
            {
                GgmlBasicOps.MoEFFNPrefillSwiGLU(
                    hiddenState, output,
                    seqLen, hiddenDim, _expertFfnLength, _numExperts, _numExpertsUsed,
                    selectedExperts, routingWeights,
                    gateW.Data, gateW.GgmlType, gateW.PerExpertNe0, gateW.PerExpertNe1, gateW.TotalRawBytes,
                    upW.Data,   upW.GgmlType,   upW.PerExpertNe0,   upW.PerExpertNe1,   upW.TotalRawBytes,
                    downW.Data, downW.GgmlType, downW.PerExpertNe0, downW.PerExpertNe1, downW.TotalRawBytes,
                    gateBias, upBias, downBias,
                    useSwiGLUOAI: true,
                    oaiAlpha: SiluAlpha,
                    oaiLimit: SiluLimit);
                InvalidateTensorDeviceCache(output);
                return true;
            }
            catch (NotSupportedException)
            {
                return false;
            }
        }

        private unsafe (float[] routingWeights, int[] selectedExperts) MoERoute(
            Tensor input, string routerWeightName, string routerBiasName, int seqLen)
        {
            using var routerScores = LinearForwardWithBias(input, routerWeightName, routerBiasName);

            float* scoresPtr = GetFloatPtr(routerScores);
            int numExperts = (int)routerScores.Sizes[1];
            int nUsed = _numExpertsUsed;
            int needed = seqLen * nUsed;

            float[] routingWeights = _moeRoutingWeightsScratch;
            int[] selectedExperts = _moeSelectedExpertsScratch;
            if (routingWeights == null || routingWeights.Length < needed)
                routingWeights = _moeRoutingWeightsScratch = new float[needed];
            if (selectedExperts == null || selectedExperts.Length < needed)
                selectedExperts = _moeSelectedExpertsScratch = new int[needed];
            int[] topK = _moeTopKScratch;

            for (int s = 0; s < seqLen; s++)
            {
                float* row = scoresPtr + s * numExperts;
                int rowOff = s * nUsed;

                // O(n*k) top-k selection (replaces the prior O(k^2*n) loop with
                // an inline 'alreadySelected' scan).
                TensorComputePrimitives.SelectTopKInPlace(row, numExperts, nUsed, topK);

                // Gather selected logits.
                float maxVal = float.NegativeInfinity;
                for (int k = 0; k < nUsed; k++)
                {
                    int idx = topK[k];
                    float v = row[idx];
                    selectedExperts[rowOff + k] = idx;
                    routingWeights[rowOff + k] = v;
                    if (v > maxVal) maxVal = v;
                }

                // Softmax-over-selected (numerically stable).
                float sumExp = 0f;
                for (int k = 0; k < nUsed; k++)
                {
                    float ex = MathF.Exp(routingWeights[rowOff + k] - maxVal);
                    routingWeights[rowOff + k] = ex;
                    sumExp += ex;
                }
                if (sumExp > 0f)
                {
                    float invSum = 1f / sumExp;
                    for (int k = 0; k < nUsed; k++)
                        routingWeights[rowOff + k] *= invSum;
                }
            }

            return (routingWeights, selectedExperts);
        }

        private unsafe Tensor ExpertFFN(Tensor input, string gateUpWeightName, string gateUpBiasName,
            string downWeightName, string downBiasName, int seqLen)
        {
            Tensor gateUp = LinearForwardWithBias(input, gateUpWeightName, gateUpBiasName);
            int halfDim = (int)(gateUp.Sizes[1] / 2);

            if (_backend == BackendType.Cuda)
            {
                var activatedCuda = new Tensor(_allocator, DType.Float32, seqLen, halfDim);
                if (CudaFusedOps.TrySwiGluOaiSplit(activatedCuda, gateUp, halfDim, SiluAlpha, SiluLimit))
                {
                    gateUp.Dispose();
                    Tensor downCuda = LinearForwardWithBias(activatedCuda, downWeightName, downBiasName);
                    activatedCuda.Dispose();
                    return downCuda;
                }
                activatedCuda.Dispose();
            }

            float* guPtr = GetFloatPtr(gateUp);

            for (int s = 0; s < seqLen; s++)
            {
                float* gatePtr = guPtr + (long)s * halfDim * 2;
                float* upPtr = gatePtr + halfDim;
                ApplySwiGluOaiInPlace(gatePtr, upPtr, halfDim);
            }

            Tensor activated;
            if (seqLen == 1)
            {
                activated = gateUp.Narrow(1, 0, halfDim);
                gateUp.Dispose();
            }
            else
            {
                using var gView = gateUp.Narrow(1, 0, halfDim);
                activated = Ops.NewContiguous(gView);
                gateUp.Dispose();
            }

            Tensor down = LinearForwardWithBias(activated, downWeightName, downBiasName);
            activated.Dispose();
            return down;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ApplySwiGluOaiInPlace(float* gate, float* up, int n)
        {
            int vLen = Vector<float>.Count;
            var vAlpha = new Vector<float>(SiluAlpha);
            var vNegAlpha = new Vector<float>(-SiluAlpha);
            var vLimit = new Vector<float>(SiluLimit);
            var vNegLimit = new Vector<float>(-SiluLimit);
            var vOne = Vector<float>.One;

            int i = 0;
            for (; i <= n - vLen; i += vLen)
            {
                var gRaw = TensorComputePrimitives.LoadVector(gate + i);
                var uRaw = TensorComputePrimitives.LoadVector(up + i);

                var x = Vector.Min(gRaw, vLimit);
                var y = Vector.Max(Vector.Min(uRaw, vLimit), vNegLimit);

                var negAx = x * vNegAlpha;
                var expNegAx = VecExpApprox(negAx);
                var sigmoid = vOne / (vOne + expNegAx);
                var result = x * sigmoid * (y + vOne);

                TensorComputePrimitives.StoreVector(gate + i, result);
            }

            for (; i < n; i++)
            {
                float x = MathF.Min(gate[i], SiluLimit);
                float y = Math.Clamp(up[i], -SiluLimit, SiluLimit);
                float outGlu = x / (1.0f + MathF.Exp(SiluAlpha * (-x)));
                gate[i] = outGlu * (y + 1.0f);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector<float> VecExpApprox(Vector<float> x)
        {
            var clampLo = new Vector<float>(-88.0f);
            var clampHi = new Vector<float>(88.0f);
            x = Vector.Max(x, clampLo);
            x = Vector.Min(x, clampHi);

            var ln2inv = new Vector<float>(1.4426950409f);
            var n = x * ln2inv;

            var half = new Vector<float>(0.5f);
            var nFloor = Vector.Floor(n + half);

            var ln2 = new Vector<float>(0.6931471806f);
            var r = x - nFloor * ln2;

            var c0 = Vector<float>.One;
            var c1 = Vector<float>.One;
            var c2 = new Vector<float>(0.5f);
            var c3 = new Vector<float>(0.16666667f);
            var c4 = new Vector<float>(0.04166667f);
            var c5 = new Vector<float>(0.00833333f);

            var poly = c0 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r * c5))));

            var intN = Vector.ConvertToInt32(nFloor);
            var bias = new Vector<int>(127);
            var shift = intN + bias;
            var pow2 = Vector.AsVectorSingle(shift << 23);

            return poly * pow2;
        }

        #endregion

        #region Linear with Bias

        private unsafe Tensor LinearForwardWithBias(Tensor input, string weightName, string biasName)
        {
            Tensor result = LinearForward(input, weightName);
            if (result == null)
                return null;

            if (_weights.TryGetValue(biasName, out var bias))
            {
                if (_backend == BackendType.Cuda && CudaFusedOps.TryAddBiasRows(result, bias))
                    return result;

                int seqLen = (int)result.Sizes[0];
                int outDim = (int)result.Sizes[1];
                float* rPtr = GetFloatPtr(result);
                float* bPtr = GetFloatPtr(bias);
                int biasDim = (int)bias.ElementCount();
                int dim = Math.Min(outDim, biasDim);

                for (int s = 0; s < seqLen; s++)
                    VecScaleAdd(rPtr + (long)s * outDim, bPtr, 1.0f, dim);
            }

            return result;
        }

        #endregion

        public override void Dispose()
        {
            if (_kvCacheK != null)
                foreach (var t in _kvCacheK) t?.Dispose();
            if (_kvCacheV != null)
                foreach (var t in _kvCacheV) t?.Dispose();
            if (_layerSinksMlx != null)
                foreach (var t in _layerSinksMlx) t?.Dispose();
            if (_sinksHandles != null)
                foreach (var handle in _sinksHandles)
                    if (handle.IsAllocated)
                        handle.Free();
            base.Dispose();
        }
    }
}
