// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Globalization;

namespace TensorSharp.Models
{
    /// <summary>
    /// Qwen3-VL (GGUF <c>qwen3vl</c> dense, <c>qwen3vlmoe</c> mixture of experts) on the shared
    /// decoder runtime, which serves its single-sequence and batched forwards, tensor parallelism,
    /// layer split, their hybrid and expert offload. Matches llama.cpp's qwen3vl / qwen3vl-moe text
    /// graphs and the Hugging Face <c>Qwen3VLMoeTextModel</c>:
    /// <list type="bullet">
    /// <item>attention: separate Q/K/V projections (the runtime fuses them), per-head Q/K RMSNorm
    /// BEFORE a full-width NeoX rotary, the interleaved M-RoPE of
    /// <c>rope.dimension_sections</c> (T, H, W) for media rows, GQA, no output gate;</item>
    /// <item>feed-forward: SwiGLU (dense) or softmax top-K routed experts renormalized over the
    /// chosen K, with no shared expert (the runtime's MoE path);</item>
    /// <item>DeepStack: the vision tower also projects three intermediate blocks; their rows are
    /// added to the residual stream at the media rows after decoder layers 0, 1 and 2
    /// (<c>n_deepstack_layers</c>). See Qwen3VLModel.Graph.cs.</item>
    /// </list>
    /// </summary>
    public sealed partial class Qwen3VLModel : StandardDecoderModel
    {
        private readonly RotarySpec _rotary;
        private readonly int[] _ropeSections;
        private readonly int _deepStackLayers;

        public Qwen3VLModel(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null,
            int layerSplitDegree = 1)
            : base(ggufPath, backend, tpDegree, tpGroup, layerSplitDegree)
        {
            string arch = _gguf.GetString("general.architecture") ?? "qwen3vlmoe";
            Config = new ModelConfig { Architecture = arch };
            ParseBaseConfig();

            int keyLen = Config.KeyLength > 0 ? Config.KeyLength : Config.HeadDim;
            int valueLen = Config.ValueLength > 0 ? Config.ValueLength : keyLen;
            if (keyLen != valueLen)
            {
                Dispose();
                throw new NotSupportedException($"{arch} expects equal key/value head dims, got key={keyLen} value={valueLen}.");
            }

            int ropeDim = (int)_gguf.GetUint32($"{arch}.rope.dimension_count", (uint)keyLen);
            _ropeSections = _gguf.GetInt32Array($"{arch}.rope.dimension_sections") is { Length: >= 3 } sections ? sections : null;
            if (_ropeSections != null)
            {
                int pairs = 0;
                foreach (int s in _ropeSections) pairs += s;
                if (pairs * 2 != ropeDim)
                {
                    Dispose();
                    throw new NotSupportedException(
                        $"{arch}: M-RoPE sections [{string.Join(", ", _ropeSections)}] rotate {pairs * 2} dims, the head rotates {ropeDim}.");
                }
            }
            _rotary = new RotarySpec(ropeDim, neoX: true, Config.RopeBase, 1.0f / Config.RopeScale);
            _deepStackLayers = (int)_gguf.GetUint32($"{arch}.n_deepstack_layers", 0);
            if (_deepStackLayers < 0 || _deepStackLayers > Config.NumLayers)
            {
                Dispose();
                throw new NotSupportedException($"{arch}: {_deepStackLayers} DeepStack layers for a {Config.NumLayers}-layer stack.");
            }
            ParseTokenizer();

            Console.WriteLine($"Model: {arch}, Layers={Config.NumLayers}, Hidden={Config.HiddenSize}, " +
                $"Heads={Config.NumHeads}, KVHeads={Config.NumKVHeads}, HeadDim={keyLen}, Vocab={Config.VocabSize}");
            Console.WriteLine($"  RoPE dims={ropeDim} base={Config.RopeBase} (NeoX, after QK-norm)" +
                (_ropeSections != null ? $", M-RoPE sections [{string.Join(", ", _ropeSections)}]" : "") +
                (_deepStackLayers > 0 ? $", DeepStack into the first {_deepStackLayers} layers" : ""));

            InitializeDecoder(keyLen, valueLen, string.Create(CultureInfo.InvariantCulture,
                $"qwen3vl|{Config.NumHeads}|{Config.NumKVHeads}|{keyLen}|{Config.Eps:R}|{Config.RopeBase:R}|{Config.RopeScale:R}|{ropeDim}|{_deepStackLayers}"));

            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (OwnsLayer(l) && (!_weights.ContainsKey($"blk.{l}.attn_q_norm.weight") || !_weights.ContainsKey($"blk.{l}.attn_k_norm.weight")))
                {
                    Dispose();
                    throw new InvalidOperationException($"{arch} layer {l} is missing attn_q_norm / attn_k_norm weights.");
                }
            }
        }

        protected override int[] RotarySections => _ropeSections;

        protected override string FingerprintExtras => $"|qwen3vl|rope={_rotary.Dims}|deepstack={_deepStackLayers}";

        protected override IEnumerable<string> ReplicatedLayerWeights(int layer)
            => new[] { $"blk.{layer}.attn_q_norm.weight", $"blk.{layer}.attn_k_norm.weight" };

        /// <summary>Per-head Q/K RMSNorm, then the NeoX rotary (Qwen's order; Hunyuan's is the reverse).</summary>
        protected override void RotateQueryKey(RankScope rank, int layer, ref Tensor q, ref Tensor k, TokenPositions positions)
        {
            string p = $"blk.{layer}.";
            bool hostDecode = positions.Count == 1 && positions.IsContiguous && HostDecodeOps;
            HeadNorm(q, Replica(_weights[p + "attn_q_norm.weight"], rank.Stage, rank.Rank), rank.LocalHeads, positions.Count, hostDecode);
            HeadNorm(k, Replica(_weights[p + "attn_k_norm.weight"], rank.Stage, rank.Rank), rank.LocalKvHeads, positions.Count, hostDecode);
            Rotate(q, rank.LocalHeads, KeyDim, positions, _rotary);
            Rotate(k, rank.LocalKvHeads, KeyDim, positions, _rotary);
        }

        private void HeadNorm(Tensor data, Tensor alpha, int heads, int tokens, bool onHost)
        {
            if (onHost)
            {
                RMSNormInPlaceCpu(data, alpha, heads, KeyDim, Config.Eps);
                HostWritten(data);
                return;
            }
            using Tensor rows = data.View(tokens * heads, KeyDim);
            Ops.RMSNorm(rows, rows, alpha, null, Config.Eps);
        }
    }
}
