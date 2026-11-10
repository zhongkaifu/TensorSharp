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
using System.Globalization;

namespace TensorSharp.Models
{
    /// <summary>
    /// Mistral 3, on the shared decoder runtime (which provides the single-sequence and batched
    /// forwards, tensor parallelism, layer split - local or multi-node - and their hybrid).
    /// What is Mistral's own:
    /// - GPT-J (norm) style RoPE with YaRN scaling for extended context
    /// - Position-dependent Q scaling: q *= (1 + beta * log(1 + floor(pos / orig_ctx)))
    /// - No per-head QK normalization
    /// - Multimodal (vision) through the separate Pixtral vision encoder
    /// </summary>
    public partial class Mistral3Model : StandardDecoderModel
    {
        private readonly int _ropeDim;

        // YaRN scaling parameters
        private readonly float _ropeScalingBeta;
        private readonly int _ropeOrigCtx;
        private readonly float _ropeExtFactor;
        private readonly float _ropeBetaFast;
        private readonly float _ropeBetaSlow;
        private readonly float _ropeMscale;
        private readonly float _ropeMscaleAllDim;
        private readonly float _ropeYarnLogMul;
        private readonly float _ropeAttnFactor;
        private readonly string _ropeType;
        private readonly RotarySpec _rotary;

        // Vision support
        private Mistral3VisionEncoder _visionEncoder;
        private readonly List<(Tensor embeddings, int position)> _pendingVisionEmbeddingsList = new();

        // Preserve the CLR signature used by existing compiled callers.
        public Mistral3Model(string ggufPath, BackendType backend, int tpDegree, ITensorParallelGroup tpGroup)
            : this(ggufPath, backend, tpDegree, tpGroup, 1)
        {
        }

        public Mistral3Model(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null,
            int layerSplitDegree = 1)
            : base(ggufPath, backend, tpDegree, tpGroup, layerSplitDegree)
        {
            // Hyperparameters live under whatever prefix the converter wrote: "mistral3"
            // for current conversions, "llama" for Mistral Small 3.x files converted before
            // llama.cpp had a mistral3 architecture (the registry admits those only through
            // Mistral3Architecture.IsLlamaLabelledMistral3). The protocol id is always
            // mistral3, so rendering, parsing and capabilities select this family either way.
            string arch = _gguf.GetString("general.architecture") ?? "mistral3";
            Config = new ModelConfig { Architecture = "mistral3" };
            if (!string.Equals(arch, Config.Architecture, StringComparison.Ordinal))
            {
                MetadataArchitecture = arch;
                Console.WriteLine($"  GGUF labelled '{arch}' is served as mistral3 (metadata read under '{arch}.*').");
            }
            ParseBaseConfig();

            int keyLen = Config.KeyLength > 0 ? Config.KeyLength : Config.HeadDim;
            int valueLen = Config.ValueLength > 0 ? Config.ValueLength : keyLen;
            _ropeDim = (int)_gguf.GetUint32($"{arch}.rope.dimension_count", (uint)keyLen);

            // YaRN parameters
            _ropeType = _gguf.GetString($"{arch}.rope.scaling.type", "");
            _ropeScalingBeta = _gguf.GetFloat32($"{arch}.attention.temperature_scale",
                               _gguf.GetFloat32($"{arch}.rope.scaling_beta", 0.1f));
            _ropeOrigCtx = (int)_gguf.GetUint32($"{arch}.rope.scaling.original_context_length", 0);
            Config.OriginalContextLength = _ropeOrigCtx;
            _ropeExtFactor = _gguf.GetFloat32($"{arch}.rope.scaling.extrapolation_factor", 1.0f);
            _ropeBetaFast = _gguf.GetFloat32($"{arch}.rope.scaling.yarn_beta_fast",
                            _gguf.GetFloat32($"{arch}.rope.scaling.beta_fast", 32.0f));
            _ropeBetaSlow = _gguf.GetFloat32($"{arch}.rope.scaling.yarn_beta_slow",
                            _gguf.GetFloat32($"{arch}.rope.scaling.beta_slow", 1.0f));
            _ropeMscale = _gguf.GetFloat32($"{arch}.rope.scaling.mscale", 0f);
            _ropeMscaleAllDim = _gguf.GetFloat32($"{arch}.rope.scaling.mscale_all_dim", 0f);
            _ropeYarnLogMul = _gguf.GetFloat32($"{arch}.rope.scaling.yarn_log_multiplier", 0f);
            _ropeAttnFactor = YarnAttnFactor(_gguf.GetFloat32($"{arch}.rope.scaling.attn_factor", 1f));

            Console.WriteLine($"Model: {arch}, Layers={Config.NumLayers}, Hidden={Config.HiddenSize}, " +
                $"Heads={Config.NumHeads}, KVHeads={Config.NumKVHeads}, KeyLen={keyLen}, " +
                $"ValLen={valueLen}, Vocab={Config.VocabSize}");
            Console.WriteLine($"RoPE base={Config.RopeBase}, scale={Config.RopeScale}, type={_ropeType}, " +
                $"dim={_ropeDim}, origCtx={_ropeOrigCtx}");
            if (_ropeType == "yarn")
                Console.WriteLine($"YaRN beta={_ropeScalingBeta}, betaFast={_ropeBetaFast}, " +
                    $"betaSlow={_ropeBetaSlow}, extFactor={_ropeExtFactor}, attnFactor={_ropeAttnFactor}");

            bool yarn = _ropeType == "yarn";
            _rotary = new RotarySpec(_ropeDim, neoX: false, Config.RopeBase, 1.0f / Config.RopeScale, _ropeOrigCtx,
                yarn ? _ropeExtFactor : 0f, _ropeAttnFactor, yarn ? _ropeBetaFast : 0f, yarn ? _ropeBetaSlow : 0f);

            ParseTokenizer();
            // "rope2": every position rotates through RoPEEx with the attention factor above.
            // Prefix checkpoints saved by the earlier rotary (host decode frequencies, YaRN
            // mscale not cancelled) hold different K rows and must not be reused.
            InitializeDecoder(keyLen, valueLen, string.Create(CultureInfo.InvariantCulture,
                $"mistral3|rope2|{Config.NumHeads}|{Config.NumKVHeads}|{keyLen}|{valueLen}|{Config.Eps:R}|{Config.RopeBase:R}|{Config.RopeScale:R}|{_ropeType}|{_ropeDim}|{_ropeOrigCtx}|{_ropeScalingBeta:R}|{_ropeExtFactor:R}|{_ropeBetaFast:R}|{_ropeBetaSlow:R}|{_ropeAttnFactor:R}"));
        }

        // Text LoRA: on a single device every projection of a decoder layer and the LM head
        // run through LinearForward, attn_q/k/v and ffn_gate/up as their fused weights. The one
        // fused graph that skips it (TryFusedDenseSwiGLUFFNInto) declines while an adapter is
        // selected. Gate and up never run as two weights here, so they count only fused.
        private static readonly HashSet<string> s_textLoraProjections = new(StringComparer.Ordinal)
            { "attn_q", "attn_k", "attn_v", "attn_qkv", "attn_output", "ffn_gate_up", "ffn_down" };

        protected override bool SupportsTextLora => true;

        protected override string? TextLoraWeightRefusal(string weightName)
            => weightName == "output.weight" || IsDecoderLayerProjection(weightName, s_textLoraProjections)
                ? null
                : "is not one of the projections Mistral 3 applies a LoRA delta to";

        // ---------------------------------------------------------------- vision

        public void LoadVisionEncoder(string mmProjPath)
        {
            RefuseVisionOffFirstStage();
            _visionEncoder = new Mistral3VisionEncoder(mmProjPath, _allocator);
            _visionEncoder.SetHostModel(this);
        }

        /// <summary>Text-embedding rows for <paramref name="tokens"/>, for multimodal
        /// layouts that interleave marker tokens with encoder output.</summary>
        internal Tensor EmbedTokensForMultimodal(int[] tokens) => Embedding(tokens);

        public void SetVisionEmbeddings(Tensor embeddings, int insertPosition)
        {
            RefuseVisionOffFirstStage();
            _pendingVisionEmbeddingsList.Add((embeddings, insertPosition));
        }

        /// <summary>Image rows are embedded and injected where this node embeds
        /// tokens. In a multi-node pipeline that runs the last stage on node 0 (the
        /// default), the first stage is another node, so images are refused rather
        /// than dropped.</summary>
        private void RefuseVisionOffFirstStage()
        {
            if (LayerSplitGroup?.NodeCount > 1 && LayerPlacement != null && !LayerPlacement.IsFirst)
                throw new NotSupportedException(
                    "Multi-node layer split runs node 0 as the last stage, so it cannot inject image embeddings. " +
                    $"Set {LayerSplitPlan.DriverStageVariable}=first on every node to serve images.");
        }

        public Mistral3VisionEncoder VisionEncoder => _visionEncoder;

        protected override bool HasPendingMedia => _pendingVisionEmbeddingsList.Count > 0;

        protected override unsafe void InjectPendingMedia(Tensor hidden, int absoluteStart)
        {
            if (_pendingVisionEmbeddingsList.Count == 0)
                return;
            int dim = Config.HiddenSize;
            float* hPtr = GetFloatPtr(hidden);
            foreach (var (embeddings, insertPos) in _pendingVisionEmbeddingsList)
            {
                int numVisionTokens = (int)embeddings.Sizes[0];
                float* vPtr = GetFloatPtr(embeddings);
                for (int t = 0; t < numVisionTokens; t++)
                {
                    Buffer.MemoryCopy(vPtr + (long)t * dim, hPtr + (long)(insertPos + t) * dim,
                        dim * sizeof(float), dim * sizeof(float));
                }
                // insertPos is the offset within the current forward; absoluteStart makes the
                // absolute position explicit so the log reads monotonically across chunks.
                Console.WriteLine($"Injected {numVisionTokens} vision tokens at chunk-offset {insertPos} (absolute position {absoluteStart + insertPos})");
                embeddings.Dispose();
            }
            _pendingVisionEmbeddingsList.Clear();
        }

        // ---------------------------------------------------------------- rotary

        /// <summary>GPT-J (norm) RoPE with Mistral's YaRN parameters (the runtime's rotary:
        /// the same function for prompts, batches and decode), then the YaRN position-dependent
        /// Q scale.</summary>
        protected override void RotateQueryKey(RankScope rank, int layer, ref Tensor q, ref Tensor k, TokenPositions positions)
        {
            Rotate(q, rank.LocalHeads, KeyDim, positions, _rotary);
            Rotate(k, rank.LocalKvHeads, KeyDim, positions, _rotary);
            if (_ropeOrigCtx > 0)
                ApplyPositionScale(q, rank.LocalHeads * KeyDim, positions);
        }

        /// <summary>
        /// The rotary attention factor, as llama.cpp computes it (llama_context, YaRN). With
        /// a non-zero extrapolation factor the rotary multiplies cos/sin by
        /// <c>1 + 0.1 ln(factor)</c> itself, so the factor cancels that and applies the
        /// checkpoint's own magnitude scale instead: <c>mscale(factor, m) / mscale(factor,
        /// m_all)</c>, where <c>yarn_log_multiplier</c> (or <c>mscale_all_dim</c>) is
        /// <c>m_all</c> and <c>m</c> is <c>mscale</c> (default 1); without either it is the
        /// standard YaRN <c>mscale(factor, 1)</c>. Ministral 3 ships
        /// <c>yarn_log_multiplier = 1</c>, so its rotary keeps unit magnitude.
        /// </summary>
        private float YarnAttnFactor(float checkpointAttnFactor)
        {
            if (_ropeType != "yarn" || _ropeExtFactor == 0f)
                return checkpointAttnFactor;
            float factor = Config.RopeScale;
            static float Mscale(float scale, float m) => scale <= 1f ? 1f : 0.1f * m * MathF.Log(scale) + 1f;
            float mAll = _ropeYarnLogMul != 0f ? _ropeYarnLogMul : _ropeMscaleAllDim;
            float m = _ropeMscale != 0f ? _ropeMscale : 1f;
            float attn = mAll != 0f ? Mscale(factor, m) / Mscale(factor, mAll) : Mscale(factor, 1f);
            if (factor > 1f)
                attn *= 1f / (1f + 0.1f * MathF.Log(factor));
            return attn * checkpointAttnFactor;
        }

        /// <summary>Position-dependent Q scaling for YaRN, one factor per token (it scales the
        /// whole row): q *= (1 + beta * log(1 + floor(pos / orig_ctx))).</summary>
        /// <remarks>Below the original context every factor is 1, so the query is not touched at
        /// all: reading it back would cost a device round trip per layer on GPU backends.</remarks>
        private unsafe void ApplyPositionScale(Tensor q, int rowWidth, TokenPositions positions)
        {
            float* ptr = null;
            for (int t = 0; t < positions.Count; t++)
            {
                float interval = MathF.Floor((float)positions[t] / _ropeOrigCtx);
                float posScale = 1.0f + _ropeScalingBeta * MathF.Log(1.0f + interval);
                if (MathF.Abs(posScale - 1.0f) < 1e-7f) continue;
                if (ptr == null) ptr = GetFloatPtr(q);
                VecScale(ptr + (long)t * rowWidth, posScale, rowWidth);
            }
            if (ptr != null)
                HostWritten(q);
        }

        public override void Dispose()
        {
            _visionEncoder?.Dispose();
            foreach (var (embeddings, _) in _pendingVisionEmbeddingsList)
                embeddings?.Dispose();
            _pendingVisionEmbeddingsList.Clear();
            base.Dispose();
        }
    }
}
