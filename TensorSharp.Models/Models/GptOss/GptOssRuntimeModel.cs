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
    /// GPT-OSS (GGUF <c>gpt-oss</c> / <c>gptoss</c>: gpt-oss-20b, gpt-oss-120b) on the shared decoder runtime, which provides
    /// every placement (one device, <c>--tp</c>, whole-layer stages), continuous batching, the prefix cache, checkpoints and
    /// speculation (n-gram drafts, DFlash drafters); its steps run as whole-forward graphs on every executor
    /// (<see cref="StandardDecoderModel.RequiresGraphs"/>). The recipe (llama.cpp <c>src/models/openai-moe-iswa.cpp</c>):
    /// <list type="bullet">
    /// <item>attention on every layer, sliding-window on the even layers (keys (p - window, p], <c>attention.sliding_window</c>,
    /// 128) and full causal on the odd ones (llama.cpp's <c>set_swa_pattern(2)</c>);</item>
    /// <item>Q, K and V with biases (fused into one product and one bias at load), NeoX rotary with YaRN scaling
    /// (<c>rope.scaling.factor</c>, <c>original_context_length</c>, beta_fast 32, beta_slow 1), no Q/K norm, GQA at
    /// 1/sqrt(head_dim), a learned per-head sink logit in every query's softmax (<c>attn_sinks.weight</c>), and the output
    /// projection's bias;</item>
    /// <item>routed experts on every layer after <c>post_attention_norm</c>: the router's logits plus its bias, the top k of the
    /// logits, the softmax over those k as the route weights (llama.cpp's SOFTMAX_WEIGHT gating); per expert gate and up
    /// products with their biases (ggml_add_id), the clamped SwiGLU g * sigmoid(1.702 g) * (u + 1) with g = min(gate, 7) and
    /// u = clamp(up, -7, 7) (ggml_swiglu_oai), the down product with its bias, the weighted sum;</item>
    /// <item>the head: <c>output_norm</c> and <c>output</c>.</item>
    /// </list>
    /// Text only. The Harmony chat format and its output parser are the architecture's (ChatTemplate, HarmonyOutputParser).
    /// </summary>
    public sealed partial class GptOssRuntimeModel : StandardDecoderModel
    {
        /// <summary>The clamped SwiGLU's constants (llama.cpp's LLM_FFN_SWIGLU_OAI_MOE; the reference implementation's).</summary>
        internal const float SwigluAlpha = 1.702f, SwigluLimit = 7.0f;

        private const int DefaultSlidingWindow = 128;

        private static readonly string[] RequiredLayerWeights =
        {
            "attn_norm.weight", "attn_sinks.weight", "attn_output.weight", "attn_output.bias", "post_attention_norm.weight",
            "ffn_gate_inp.weight", "ffn_gate_inp.bias", "ffn_gate_exps.bias", "ffn_up_exps.bias", "ffn_down_exps.bias",
        };

        private readonly int _headDim;
        private readonly int _window;
        private readonly int _experts;
        private readonly int _expertsUsed;
        private readonly int _expertWidth;
        private readonly RotarySpec _rotary;

        public GptOssRuntimeModel(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null,
            int layerSplitDegree = 1, string draftModelPath = null)
            : base(ggufPath, backend, tpDegree, tpGroup, layerSplitDegree)
        {
            string arch = _gguf.GetString("general.architecture") ?? "gpt-oss";
            Config = new ModelConfig { Architecture = arch };
            ParseBaseConfig();

            _headDim = Config.KeyLength > 0 ? Config.KeyLength : Config.HeadDim;
            int valueDim = Config.ValueLength > 0 ? Config.ValueLength : _headDim;
            if (valueDim != _headDim)
                throw new NotSupportedException($"{arch}: key width {_headDim} != value width {valueDim}.");
            _window = (int)_gguf.GetUint32($"{arch}.attention.sliding_window", DefaultSlidingWindow);
            _experts = (int)_gguf.GetUint32($"{arch}.expert_count", 0);
            _expertsUsed = (int)_gguf.GetUint32($"{arch}.expert_used_count", 0);
            _expertWidth = (int)_gguf.GetUint32($"{arch}.expert_feed_forward_length", (uint)Config.IntermediateSize);
            if (_experts <= 0 || _expertsUsed <= 0 || _expertsUsed > _experts)
                throw new InvalidOperationException($"{arch}: routed experts need expert_count and expert_used_count (got {_experts} and {_expertsUsed}).");
            Config.NumExperts = _experts;
            Config.NumExpertsUsed = _expertsUsed;
            Config.SlidingWindow = _window;
            Config.OriginalContextLength = (int)_gguf.GetUint32($"{arch}.rope.scaling.original_context_length", 4096);
            // The experts' width: the runtime sizes and shards the feed-forwards by it.
            Config.IntermediateSize = _expertWidth;

            // NeoX rotary with YaRN over the whole head (ggml_rope_ext: ext_factor 1, attn_factor 1, the GGUF's betas).
            float betaFast = _gguf.GetFloat32($"{arch}.rope.scaling.yarn_beta_fast", 32f);
            float betaSlow = _gguf.GetFloat32($"{arch}.rope.scaling.yarn_beta_slow", 1f);
            string scaling = _gguf.GetString($"{arch}.rope.scaling.type") ?? "yarn";
            bool yarn = string.Equals(scaling, "yarn", StringComparison.OrdinalIgnoreCase) && Config.RopeScale > 1f;
            _rotary = new RotarySpec(_headDim, neoX: true, Config.RopeBase, 1.0f / Config.RopeScale, Config.OriginalContextLength,
                yarn ? 1f : 0f, 1f, betaFast, betaSlow);
            ParseTokenizer();

            Console.WriteLine($"Model: {arch} (shared decoder runtime), Layers={Config.NumLayers} (sliding-window {_window} on the even ones), " +
                $"Hidden={Config.HiddenSize}, Heads={Config.NumHeads}, KVHeads={Config.NumKVHeads}, HeadDim={_headDim}, Vocab={Config.VocabSize}");
            Console.WriteLine($"  RoPE (NeoX{(yarn ? ", YaRN" : "")}) base={Config.RopeBase} scale={Config.RopeScale} original context={Config.OriginalContextLength}; " +
                $"MoE: {_experts} experts, top-{_expertsUsed}, width {_expertWidth}; attention sinks; biased projections");

            InitializeDecoder(_headDim, _headDim, string.Create(CultureInfo.InvariantCulture,
                $"gpt-oss|{Config.NumHeads}|{Config.NumKVHeads}|{_headDim}|{Config.Eps:R}|{Config.RopeBase:R}|{Config.RopeScale:R}|" +
                $"{Config.OriginalContextLength}|{yarn}|{betaFast:R}|{betaSlow:R}|{_window}|{_experts}|{_expertsUsed}|{_expertWidth}"));
            try
            {
                ValidateLayerWeights();
            }
            catch
            {
                Dispose();
                throw;
            }
            if (!string.IsNullOrWhiteSpace(draftModelPath) && SpeculativeDraftHeadLoader.IsDFlashDrafter(draftModelPath))
                AttachDraftModel(draftModelPath);
        }

        // ---------------------------------------------------------------- the runtime's declarations

        /// <summary>The sinks, the biases and the clamped SwiGLU experts are lowered to graphs only.</summary>
        protected override bool RequiresGraphs => true;

        protected override int SlidingWindow(int layer) => layer % 2 == 0 ? _window : 0;

        protected override string AttentionSinksWeight(int layer) => LayerName(layer, "attn_sinks.weight");

        protected override bool AttentionProjectionsHaveBiases => true;

        /// <summary>The routed experts read <c>post_attention_norm</c>.</summary>
        protected override string FfnNormWeight(int layer) => LayerName(layer, "post_attention_norm.weight");

        protected override string FingerprintExtras =>
            string.Create(CultureInfo.InvariantCulture,
                $"|gpt-oss|swa={_window}:even|rope={_rotary?.Dims}@{_rotary?.FreqBase:R}/neox/yarn{_rotary?.ExtFactor:R}|moe={_experts}x{_expertsUsed}/oai");

        /// <summary>
        /// A q4_0 cache costs GPT-OSS too much (gpt-oss-20b MXFP4 on ggml_cuda, an A40, 384 tokens of the README after 64:
        /// mean NLL 2.3060 at f16, 2.3056 at q8_0, 2.7811 at q4_0), and the tensor executors' graphs do not run its
        /// sliding-window rings over a block-quantized cache (direct CUDA's q8_0 row writes do not wrap): q8_0 stays on the
        /// GGML backends, every other request becomes f16, as the legacy model made every block type.
        /// </summary>
        protected override string BlockQuantizedKvCacheRefusal(KvCacheDtype requested) => requested switch
        {
            KvCacheDtype.Q4_0 => "keeps an f16 K/V cache for a q4_0 request: its attention loses 0.48 nat a token over a q4_0 cache",
            KvCacheDtype.Q8_0 when !IsGgmlBackend =>
                "runs its sliding-window rings over a q8_0 K/V cache on the GGML backends only",
            _ => null,
        };

        /// <summary>Not implemented: GPT-OSS's blocks run in whole-forward graphs only.</summary>
        protected override void RotateQueryKey(RankScope rank, int layer, ref Tensor q, ref Tensor k, TokenPositions positions)
            => throw new NotSupportedException($"{Config.Architecture} runs whole-forward graphs only.");

        /// <summary>
        /// What a tensor-parallel degree must also meet: each rank keeps a slice of every expert's width, and the down
        /// projection's rows split along it into whole quantization blocks (MXFP4's 32 values: gpt-oss-20b's 2880-wide experts,
        /// 90 blocks a row, split over 2 ranks but not 4 or 8), as the attention output's rows split into whole blocks of the
        /// query heads' width.
        /// </summary>
        protected override IEnumerable<string> ShardingProblems(int degree)
        {
            if (_expertWidth % degree != 0) yield return $"the experts' width {_expertWidth}";
            foreach (string suffix in new[] { "attn_output.weight", "ffn_down_exps.weight" })
            {
                for (int l = 0; l < Config.NumLayers; l++)
                {
                    if (!_gguf.Tensors.TryGetValue($"blk.{l}.{suffix}", out GgufTensorInfo info) || info.Shape.Length < 1) continue;
                    long block = GgufFile.GetBlockSize(info.Type), width = (long)info.Shape[0];
                    if (block <= 1 || width % block != 0 || (width / block) % degree == 0) continue;
                    yield return $"the {width / block} quantization blocks of a {suffix[..^".weight".Length]} row ({info.Type}, layer {l})";
                    break;
                }
            }
        }

        /// <summary>Every owned layer carries the weights its blocks read: refused at load, not at the first forward.</summary>
        private void ValidateLayerWeights()
        {
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!OwnsLayer(l)) continue;
                foreach (string suffix in RequiredLayerWeights)
                {
                    string name = LayerName(l, suffix);
                    if (!HasWeight(name))
                        throw new InvalidOperationException($"{Config.Architecture} layer {l} is missing '{name}'.");
                }
                foreach (string stack in new[] { "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight" })
                    if (!_gguf.Tensors.ContainsKey(LayerName(l, stack)))
                        throw new InvalidOperationException($"{Config.Architecture} layer {l} is missing the stacked experts '{LayerName(l, stack)}'.");
            }
        }

        /// <summary>Tests: whether <paramref name="layer"/> attends a sliding window.</summary>
        internal bool IsSlidingWindowLayerForTests(int layer) => SlidingWindow(layer) > 0;

        /// <summary>Whether the model holds a weight of this name, fused or not (tests).</summary>
        internal bool HasWeightForTests(string name) => HasWeight(name);
    }
}
