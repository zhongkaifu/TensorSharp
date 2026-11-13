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
    /// Hunyuan dense transformer (GGUF <c>hunyuan-dense</c>), on the shared decoder runtime,
    /// which provides its single-sequence and batched forwards, tensor parallelism, layer split
    /// and their hybrid. Matches llama.cpp's hunyuan-vl text graph: RMSNorm -> QKV -> NeoX RoPE
    /// -> per-head Q/K RMSNorm -> GQA attention -> SwiGLU. QK-norm is AFTER RoPE, the opposite
    /// of Qwen 3.5; copying Qwen's order gives fluent but wrong output.
    /// </summary>
    public sealed partial class HunyuanDenseModel : StandardDecoderModel
    {
        private readonly int _ropeDim;
        private readonly RotarySpec _rotary;

        public HunyuanDenseModel(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null,
            int layerSplitDegree = 1)
            : base(ggufPath, backend, tpDegree, tpGroup, layerSplitDegree)
        {
            string arch = _gguf.GetString("general.architecture") ?? "hunyuan-dense";
            Config = new ModelConfig { Architecture = arch };
            ParseBaseConfig();

            int keyLen = Config.KeyLength > 0 ? Config.KeyLength : Config.HeadDim;
            int valueLen = Config.ValueLength > 0 ? Config.ValueLength : keyLen;
            if (keyLen != valueLen)
                throw new NotSupportedException($"hunyuan-dense expects equal key/value head dims, got key={keyLen} value={valueLen}.");

            _ropeDim = (int)_gguf.GetUint32($"{arch}.rope.dimension_count", (uint)keyLen);
            ApplyHunyuanRopeBase(arch, keyLen);
            _rotary = new RotarySpec(_ropeDim, neoX: true, Config.RopeBase, 1.0f / Config.RopeScale);
            ParseTokenizer();

            Console.WriteLine($"Model: {arch}, Layers={Config.NumLayers}, Hidden={Config.HiddenSize}, " +
                $"Heads={Config.NumHeads}, KVHeads={Config.NumKVHeads}, KeyLen={keyLen}, " +
                $"ValLen={valueLen}, Vocab={Config.VocabSize}");
            Console.WriteLine($"RoPE base={Config.RopeBase}, scale={Config.RopeScale}, dim={_ropeDim} (NeoX, then QK-norm)");

            InitializeDecoder(keyLen, valueLen, string.Create(CultureInfo.InvariantCulture,
                $"hunyuan-dense|{Config.NumHeads}|{Config.NumKVHeads}|{keyLen}|{Config.Eps:R}|{Config.RopeBase:R}|{Config.RopeScale:R}|{_ropeDim}"));

            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (OwnsLayer(l) && (!_weights.ContainsKey($"blk.{l}.attn_q_norm.weight") || !_weights.ContainsKey($"blk.{l}.attn_k_norm.weight")))
                {
                    Dispose();
                    throw new InvalidOperationException($"hunyuan-dense layer {l} is missing attn_q_norm / attn_k_norm weights.");
                }
            }
        }

        // Text LoRA: every projection of a decoder layer and the LM head run through
        // LinearForward, attn_q/k/v and ffn_gate/up fused or (split FFN) as two weights. The
        // fused dense FFN graph declines while an adapter is selected.
        private static readonly HashSet<string> s_textLoraProjections = new(StringComparer.Ordinal)
            { "attn_q", "attn_k", "attn_v", "attn_qkv", "attn_output", "ffn_gate", "ffn_up", "ffn_gate_up", "ffn_down" };

        protected override bool SupportsTextLora => true;

        protected override string? TextLoraWeightRefusal(string weightName)
            => weightName == "output.weight" || IsDecoderLayerProjection(weightName, s_textLoraProjections)
                ? null
                : "is not one of the projections hunyuan-dense applies a LoRA delta to";

        protected override string FingerprintExtras => $"|rope={_ropeDim}";

        protected override IEnumerable<string> ReplicatedLayerWeights(int layer)
            => new[] { $"blk.{layer}.attn_q_norm.weight", $"blk.{layer}.attn_k_norm.weight" };

        /// <summary>
        /// llama.cpp hunyuan-vl: <c>base = rope_theta * alpha^(dim / (dim - 2))</c>
        /// when XDRoPE / NTK alpha is present. Hy-MT2 Q4 ships <c>scaling.type=none</c>.
        /// </summary>
        private void ApplyHunyuanRopeBase(string arch, int keyLen)
        {
            float alpha = _gguf.GetFloat32($"{arch}.rope.scaling.alpha", 0f);
            if (alpha <= 0f || keyLen <= 2)
                return;
            Config.RopeBase *= MathF.Pow(alpha, (float)keyLen / (keyLen - 2));
            Console.WriteLine($"  Applied Hunyuan NTK RoPE alpha={alpha}, effective base={Config.RopeBase}");
        }

        /// <summary>NeoX RoPE first, then per-head Q/K RMSNorm. Do not use Qwen 3.5's fused
        /// QKNorm+RoPE kernel: that applies the norm before rotation.</summary>
        protected override void RotateQueryKey(RankScope rank, int layer, ref Tensor q, ref Tensor k, TokenPositions positions)
        {
            string p = $"blk.{layer}.";
            Rotate(q, rank.LocalHeads, KeyDim, positions, _rotary);
            Rotate(k, rank.LocalKvHeads, KeyDim, positions, _rotary);
            // The host path serves a single sequence's decode token; a paged batch, even of
            // one token, keeps the device op.
            bool hostDecode = positions.Count == 1 && positions.IsContiguous && HostDecodeOps;
            ApplyQKNorm(q, Replica(_weights[p + "attn_q_norm.weight"], rank.Stage, rank.Rank), rank.LocalHeads, positions.Count, hostDecode);
            ApplyQKNorm(k, Replica(_weights[p + "attn_k_norm.weight"], rank.Stage, rank.Rank), rank.LocalKvHeads, positions.Count, hostDecode);
        }

        private void ApplyQKNorm(Tensor data, Tensor alpha, int numHeads, int tokens, bool hostDecode)
        {
            if (hostDecode)
            {
                RMSNormInPlaceCpu(data, alpha, numHeads, KeyDim, Config.Eps);
                HostWritten(data);
                return;
            }
            using Tensor reshaped = data.View(tokens * numHeads, KeyDim);
            Ops.RMSNorm(reshaped, reshaped, alpha, null, Config.Eps);
        }
    }
}
