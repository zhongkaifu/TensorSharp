// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace TensorSharp.Models
{
    /// <summary>
    /// Gemma 4 (GGUF <c>gemma4</c>: E2B/E4B, 12B, 31B and the 26B-A4B MoE) on the shared decoder runtime, which provides every
    /// placement, continuous batching, the prefix cache, checkpoints and speculation; its steps run as whole-forward graphs on
    /// every executor (<see cref="RequiresGraphs"/>). The recipe (llama.cpp <c>src/models/gemma4.cpp</c>):
    /// <list type="bullet">
    /// <item>per-layer attention geometry: sliding-window layers (<c>attention.sliding_window_pattern</c>, keys (p - W, p]) of
    /// <c>key_length_swa</c>-wide heads, global layers of <c>key_length</c>-wide heads, each kind its own KV head count
    /// (<c>attention.head_count_kv</c> per layer, or <c>global_head_count_kv</c>);</item>
    /// <item>attention: per-head RMSNorm of Q and K (weighted), an unweighted RMSNorm of V, NeoX rotary on every dim of a local
    /// head at <c>rope.freq_base_swa</c> and on the first <c>rope.dimension_count</c> dims of a global head at
    /// <c>rope.freq_base</c> with <c>rope_freqs.weight</c>'s proportional factors; scale 1 (the norms carry 1/sqrt(d)); a global
    /// layer without <c>attn_v</c> takes its raw keys as values;</item>
    /// <item>KV sharing (E2B/E4B): the last <c>attention.shared_kv_layers</c> layers attend the cache of the last earlier layer
    /// of their kind, projecting their queries only (<see cref="KvSourceLayer"/>);</item>
    /// <item>sandwich norms: <c>post_attention_norm</c> on the attention output and <c>post_ffw_norm</c> on the feed-forward
    /// output before each residual add; a GeGLU feed-forward (gelu_pytorch_tanh);</item>
    /// <item>MoE layers (26B-A4B): the dense GeGLU branch (<c>post_ffw_norm_1</c>) beside a routed one: the router reads the raw
    /// residual through an unweighted RMSNorm, 1/sqrt(n_embd) and <c>ffn_gate_inp.scale</c>; softmax, top-k, renormalized and
    /// times <c>ffn_down_exps.scale</c>; GeGLU experts over <c>pre_ffw_norm_2</c>(residual) (<c>post_ffw_norm_2</c>); the two
    /// branches' sum through <c>post_ffw_norm</c>;</item>
    /// <item>per-layer embeddings (E-series): <c>per_layer_token_embd</c> rows (times sqrt(ple)) plus RMSNorm of
    /// <c>per_layer_model_proj</c>(embedded) / sqrt(n_embd), times 1/sqrt(2), gate each layer's
    /// <c>proj(gelu(inp_gate(h)) * ple_l)</c> through <c>post_norm</c> into the stream;</item>
    /// <item><c>layer_output_scale</c> on each layer's output; the embedded rows times sqrt(n_embd) (media rows as they come);
    /// the head (<c>output</c>, or <c>token_embd</c> tied) softcapped at <c>final_logit_softcapping</c>.</item>
    /// </list>
    /// Media soft tokens (images, video frames, audio) attend each other in both directions within their forward
    /// (<see cref="MediaAttendsBidirectionally"/>).
    /// The recipe is shared: <see cref="Gemma4RuntimeModel"/> (the autoregressive checkpoints, with the gemma4-assistant
    /// drafter and the audio tower) and <see cref="DiffusionGemmaModel"/> (the block-diffusion checkpoint on the same backbone)
    /// derive from it.
    /// </summary>
    public abstract partial class Gemma4DecoderModel : StandardDecoderModel
    {
        private protected readonly bool[] _isLocal;
        private protected readonly int[] _kvHeads;
        private protected readonly int _window;
        private protected readonly int _localHeadDim;
        private protected readonly int _globalHeadDim;
        private readonly int _pleDim;
        private readonly int _sharedKvLayers;
        private protected readonly int[] _kvSource;
        private protected readonly float _softcap;
        private protected readonly RotarySpec _localRope;
        private protected readonly RotarySpec _globalRope;
        private readonly int _numExperts;
        private readonly int _expertsUsed;
        private protected readonly float[] _layerScale;
        private readonly bool[] _hasValue;

        private protected Gemma4DecoderModel(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null,
            int layerSplitDegree = 1)
            : base(ggufPath, backend, tpDegree, tpGroup, layerSplitDegree)
        {
            string arch = _gguf.GetString("general.architecture") ?? "gemma4";
            Config = new ModelConfig { Architecture = arch };
            ParseBaseConfig();
            int layers = Config.NumLayers;

            _window = (int)_gguf.GetUint32($"{arch}.attention.sliding_window", 512);
            Config.SlidingWindow = _window;
            _isLocal = new bool[layers];
            bool[] pattern = _gguf.GetBoolArray($"{arch}.attention.sliding_window_pattern");
            for (int l = 0; l < layers; l++)
                _isLocal[l] = pattern != null && l < pattern.Length && pattern[l];

            int globalHead = (int)_gguf.GetUint32($"{arch}.attention.key_length", 512);
            int localHead = (int)_gguf.GetUint32($"{arch}.attention.key_length_swa", 256);
            // KV heads: a per-layer array, or the scalar for local layers and global_head_count_kv for global ones.
            _kvHeads = new int[layers];
            int[] kvArray = _gguf.GetInt32Array($"{arch}.attention.head_count_kv");
            int globalKv = (int)_gguf.GetUint32($"{arch}.attention.global_head_count_kv", 0);
            int localKv = kvArray is { Length: > 0 } ? kvArray[0] : Config.NumKVHeads;
            for (int l = 0; l < layers; l++)
            {
                if (kvArray != null && l < kvArray.Length && kvArray[l] > 0) _kvHeads[l] = kvArray[l];
                else _kvHeads[l] = !_isLocal[l] && globalKv > 0 ? globalKv : localKv;
            }
            // The first local layer's KV heads are the model's (Config.NumKVHeads): draft-free defaults and the fingerprint.
            int firstLocal = Array.IndexOf(_isLocal, true);
            Config.NumKVHeads = firstLocal >= 0 ? _kvHeads[firstLocal] : _kvHeads[0];
            // The head widths as the weights have them (legacy's DetectHeadDimsFromWeights): attn_k rows over the layer's KV heads.
            _localHeadDim = HeadWidthFromWeights(local: true) ?? localHead;
            _globalHeadDim = HeadWidthFromWeights(local: false) ?? globalHead;
            int defaultHead = firstLocal >= 0 ? _localHeadDim : _globalHeadDim;

            _hasValue = new bool[layers];
            for (int l = 0; l < layers; l++)
                _hasValue[l] = _gguf.Tensors.ContainsKey($"blk.{l}.attn_v.weight");

            // Rotary: local NeoX over the whole local head; global NeoX over rope.dimension_count dims with rope_freqs' factors.
            float localBase = _gguf.GetFloat32($"{arch}.rope.freq_base_swa", 0f);
            if (localBase == 0f) localBase = _gguf.GetFloat32($"{arch}.rope.local.freq_base", 10000f);
            int localRopeDims = (int)_gguf.GetUint32($"{arch}.rope.dimension_count_swa", (uint)_localHeadDim);
            if (localRopeDims <= 0 || localRopeDims > _localHeadDim) localRopeDims = _localHeadDim;
            int globalRopeDims = (int)_gguf.GetUint32($"{arch}.rope.dimension_count", 0);
            if (globalRopeDims == 0)
                globalRopeDims = (int)(_globalHeadDim * _gguf.GetFloat32($"{arch}.rope.partial_rotary_factor", 1.0f));
            if (globalRopeDims > _globalHeadDim) globalRopeDims = _globalHeadDim;
            _localRope = new RotarySpec(localRopeDims, neoX: true, localBase);
            float[] factors = ReadRopeFactors(globalRopeDims / 2);
            _globalRope = factors != null
                ? new RotarySpec(globalRopeDims, neoX: true, Config.RopeBase, freqFactorsWeight: "rope_freqs.weight", freqFactors: factors)
                : new RotarySpec(globalRopeDims, neoX: true, Config.RopeBase);

            _softcap = _gguf.GetFloat32($"{arch}.final_logit_softcapping", 0f);
            _pleDim = (int)_gguf.GetUint32($"{arch}.embedding_length_per_layer_input", 0);
            _sharedKvLayers = (int)_gguf.GetUint32($"{arch}.attention.shared_kv_layers", 0);
            _kvSource = BuildKvSources(layers);
            _numExperts = (int)_gguf.GetUint32($"{arch}.expert_count", 0);
            _expertsUsed = (int)_gguf.GetUint32($"{arch}.expert_used_count", 0);
            _layerScale = new float[layers];
            Array.Fill(_layerScale, 1f);
            ParseTokenizer();

            int local = _isLocal.Count(x => x), shared = _kvSource.Where((s, l) => s != l).Count();
            Console.WriteLine($"Model: {arch} (shared decoder runtime{FamilyNote}), Layers={layers} ({local} sliding-window, {layers - local} global" +
                (shared > 0 ? $", {shared} sharing K/V" : "") + $"), Hidden={Config.HiddenSize}, Heads={Config.NumHeads}, " +
                $"KVHeads={string.Join("/", _kvHeads.Distinct())}, HeadDim={_localHeadDim}/{_globalHeadDim}, FFN={Config.IntermediateSize}, Vocab={Config.VocabSize}");
            Console.WriteLine($"  RoPE local base={localBase} dims={localRopeDims}; global base={Config.RopeBase} dims={globalRopeDims}" +
                (factors != null ? $" ({factors.Count(f => f == 1f)} of {factors.Length} pairs rotated by rope_freqs)" : "") +
                $"; window={_window}; Softcap={_softcap}; PLE={_pleDim}" +
                (_numExperts > 0 ? $"; MoE {_numExperts} experts, top-{_expertsUsed}" : ""));

            InitializeDecoder(defaultHead, defaultHead, string.Create(CultureInfo.InvariantCulture,
                $"gemma4|{Config.NumHeads}|{string.Join(",", _kvHeads)}|{_localHeadDim}|{_globalHeadDim}|{Config.Eps:R}|{localBase:R}|{Config.RopeBase:R}|" +
                $"{localRopeDims}|{globalRopeDims}|{_window}|{LayerBits(_isLocal)}|{_softcap:R}|{_pleDim}|{_sharedKvLayers}|{_numExperts}|{_expertsUsed}"));
            try
            {
                ValidateLayerWeights();
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        /// <summary>What the load line says beside the runtime (a derived family's own note).</summary>
        private protected virtual string FamilyNote => "";

        private protected static string LayerBits(bool[] flags) => new(flags.Select(f => f ? '1' : '0').ToArray());

        /// <summary>The head width of the local (or global) layers from the first such layer's <c>attn_k</c> rows over its KV
        /// heads, where the metadata may disagree with the weights; null without such a layer.</summary>
        private int? HeadWidthFromWeights(bool local)
        {
            for (int l = 0; l < _isLocal.Length; l++)
            {
                if (_isLocal[l] != local || _kvHeads[l] <= 0) continue;
                if (_gguf.Tensors.TryGetValue($"blk.{l}.attn_k.weight", out GgufTensorInfo k) && k.Shape.Length >= 2)
                    return (int)((long)k.Shape[1] / _kvHeads[l]);
            }
            return null;
        }

        /// <summary><c>rope_freqs.weight</c>'s first <paramref name="pairs"/> values (null without the tensor).</summary>
        private float[] ReadRopeFactors(int pairs)
        {
            if (!_gguf.Tensors.TryGetValue("rope_freqs.weight", out GgufTensorInfo info)) return null;
            long n = info.NumElements;
            if (n < pairs) throw new InvalidOperationException($"rope_freqs.weight has {n} factors, the global rotary needs {pairs}.");
            var all = new float[n];
            _gguf.ReadTensorDataToFloat32(info, all, n);
            return all.AsSpan(0, pairs).ToArray();
        }

        /// <summary>legacy's BuildKVDonorMap: each of the last <c>shared_kv_layers</c> layers attends the cache of the last layer
        /// of its kind before them.</summary>
        private int[] BuildKvSources(int layers)
        {
            var source = new int[layers];
            for (int l = 0; l < layers; l++) source[l] = l;
            if (_sharedKvLayers <= 0) return source;
            int firstShared = layers - _sharedKvLayers;
            for (int l = firstShared; l < layers; l++)
                for (int j = firstShared - 1; j >= 0; j--)
                    if (_isLocal[j] == _isLocal[l])
                    {
                        source[l] = j;
                        break;
                    }
            return source;
        }

        /// <summary>Whether <paramref name="layer"/> is a sliding-window layer (tests).</summary>
        internal bool IsLocalLayerForTests(int layer) => _isLocal[layer];

        /// <summary>The layer whose cache <paramref name="layer"/> attends (tests).</summary>
        internal int KvSourceForTests(int layer) => _kvSource[layer];

        // ---------------------------------------------------------------- the runtime's declarations

        protected override bool RequiresGraphs => true;
        protected override RotarySpec GraphRotary => _localRope;
        protected override bool MediaAttendsBidirectionally => true;
        protected override int LayerKeyDim(int layer) => _isLocal[layer] ? _localHeadDim : _globalHeadDim;
        protected override int LayerValueDim(int layer) => _isLocal[layer] ? _localHeadDim : _globalHeadDim;
        protected override int LayerKvHeads(int layer) => _kvHeads[layer];
        protected override int KvSourceLayer(int layer) => _kvSource[layer];
        /// <summary>Gemma 4 attends at scale 1: the Q/K norms carry the 1/sqrt(d).</summary>
        protected override float AttentionScale(int layer) => 1f;
        protected override int SlidingWindow(int layer) => _isLocal[layer] && _window > 0 ? _window : 0;
        protected override string MixerPostNormWeight(int layer) => FirstWeight(layer, "post_attention_norm.weight", "attn_post_norm.weight");
        protected override float FinalLogitSoftcap => _softcap;

        /// <summary>Host-computed experts run GeGLU (gelu_tanh(gate) * up) over the fused gate|up stack.</summary>
        private protected override int HostExpertActivation => 2;

        /// <summary>A KV-shared layer has no keys or values of its own to fuse; the others fuse Q/K/V where all three exist.</summary>
        protected override bool FusesQkv(int layer) => layer >= Config.NumLayers || (_kvSource[layer] == layer && _hasValue[layer]);

        protected override string FingerprintExtras => _isLocal == null ? "|gemma4" :
            string.Create(CultureInfo.InvariantCulture,
                $"|gemma4|swa={_window}:{LayerBits(_isLocal)}|hd={_localHeadDim}/{_globalHeadDim}|kv={string.Join(",", _kvHeads)}" +
                $"|src={string.Join(",", _kvSource)}|rope={_localRope.Dims}@{_localRope.FreqBase:R},{_globalRope.Dims}@{_globalRope.FreqBase:R}|ple={_pleDim}");

        /// <summary>The Q/K norms and the per-layer-embedding weights are read whole on every rank.</summary>
        protected override IEnumerable<string> ReplicatedLayerWeights(int layer)
        {
            yield return LayerName(layer, "attn_q_norm.weight");
            if (_kvSource[layer] == layer) yield return LayerName(layer, "attn_k_norm.weight");
            if (HasPle(layer)) yield return LayerName(layer, "post_norm.weight");
            if (!IsMoeLayer(layer)) yield break;
            foreach (string suffix in new[] { "pre_ffw_norm_2.weight", "post_ffw_norm_1.weight", "post_ffw_norm_2.weight", "ffn_gate_inp.scale", "ffn_down_exps.scale" })
                yield return LayerName(layer, suffix);
        }

        /// <summary>The per-op path is not implemented: Gemma 4's layers differ in head width, share caches and take their
        /// norms, rotary and per-layer embeddings in the graph.</summary>
        protected override void RotateQueryKey(RankScope rank, int layer, ref Tensor q, ref Tensor k, TokenPositions positions)
            => throw new NotSupportedException($"{Config.Architecture} runs whole-forward graphs only.");

        /// <summary>The per-layer output scalars and the routed experts' per-expert scales, read while the host copies exist.</summary>
        protected override void PrepareLoadedWeights()
        {
            for (int l = 0; l < Config.NumLayers; l++)
                if (_weights.TryGetValue(LayerName(l, "layer_output_scale.weight"), out Tensor s))
                    _layerScale[l] = s.GetElementAsFloat(0);
        }

        /// <summary>The first of <paramref name="names"/> the layer has (legacy's two spellings of a norm), or the first one.</summary>
        private protected string FirstWeight(int layer, params string[] names)
        {
            foreach (string n in names)
            {
                string full = LayerName(layer, n);
                if (_gguf.Tensors.ContainsKey(full)) return full;
            }
            return LayerName(layer, names[0]);
        }

        private protected bool IsMoeLayer(int layer) => _numExperts > 0 && _gguf.Tensors.ContainsKey($"blk.{layer}.ffn_gate_inp.weight");

        private bool HasPle(int layer) => _pleDim > 0 && _gguf.Tensors.ContainsKey($"blk.{layer}.inp_gate.weight");

        /// <summary>Every owned layer carries the weights its blocks read: refused at load, not at the first forward.</summary>
        private void ValidateLayerWeights()
        {
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!OwnsLayer(l)) continue;
                var required = new List<string> { "attn_norm.weight", "attn_q_norm.weight", "attn_output.weight", "ffn_norm.weight", "ffn_down.weight" };
                if (_kvSource[l] == l) required.Add("attn_k_norm.weight");
                foreach (string suffix in required)
                    if (!HasWeight(LayerName(l, suffix)))
                        throw new InvalidOperationException($"{Config.Architecture} layer {l} is missing '{LayerName(l, suffix)}'.");
                if (!HasWeight(LayerName(l, "attn_qkv.weight")) && !HasWeight(LayerName(l, "attn_q.weight")))
                    throw new InvalidOperationException($"{Config.Architecture} layer {l} has no query projection.");
            }
            if (_pleDim > 0)
                foreach (string name in new[] { "per_layer_token_embd.weight", "per_layer_model_proj.weight", "per_layer_proj_norm.weight" })
                    if (!HasWeight(name))
                        throw new InvalidOperationException($"{Config.Architecture} has per-layer embeddings but no '{name}'.");
        }
    }
}
