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
        private float[] _ropeFreqs;
        private readonly int _ropeDim;

        // YaRN scaling parameters
        private readonly float _ropeScalingBeta;
        private readonly int _ropeOrigCtx;
        private readonly float _ropeExtFactor;
        private readonly float _ropeBetaFast;
        private readonly float _ropeBetaSlow;
        private readonly float _ropeMscale;
        private readonly float _ropeMscaleAllDim;
        private readonly string _ropeType;

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

            Console.WriteLine($"Model: {arch}, Layers={Config.NumLayers}, Hidden={Config.HiddenSize}, " +
                $"Heads={Config.NumHeads}, KVHeads={Config.NumKVHeads}, KeyLen={keyLen}, " +
                $"ValLen={valueLen}, Vocab={Config.VocabSize}");
            Console.WriteLine($"RoPE base={Config.RopeBase}, scale={Config.RopeScale}, type={_ropeType}, " +
                $"dim={_ropeDim}, origCtx={_ropeOrigCtx}");
            if (_ropeType == "yarn")
                Console.WriteLine($"YaRN beta={_ropeScalingBeta}, betaFast={_ropeBetaFast}, " +
                    $"betaSlow={_ropeBetaSlow}, extFactor={_ropeExtFactor}");

            ParseTokenizer();
            PrecomputeRopeFrequencies();
            InitializeDecoder(keyLen, valueLen, string.Create(CultureInfo.InvariantCulture,
                $"mistral3|{Config.NumHeads}|{Config.NumKVHeads}|{keyLen}|{valueLen}|{Config.Eps:R}|{Config.RopeBase:R}|{Config.RopeScale:R}|{_ropeType}|{_ropeDim}|{_ropeOrigCtx}|{_ropeScalingBeta:R}|{_ropeExtFactor:R}|{_ropeBetaFast:R}|{_ropeBetaSlow:R}|{_ropeMscale:R}|{_ropeMscaleAllDim:R}"));
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

        private void PrecomputeRopeFrequencies()
        {
            int halfDim = _ropeDim / 2;
            float freqScale = 1.0f / Config.RopeScale;
            _ropeFreqs = new float[halfDim];
            for (int i = 0; i < halfDim; i++)
                _ropeFreqs[i] = freqScale / MathF.Pow(Config.RopeBase, (2.0f * i) / _ropeDim);
            if (_ropeType == "yarn" && _ropeOrigCtx > 0)
                ApplyYarnFreqCorrection(_ropeFreqs, halfDim);
        }

        /// <summary>
        /// Apply YaRN frequency correction to precomputed RoPE frequencies for decode path.
        /// Interpolates between extrapolated and interpolated frequencies based on
        /// whether each frequency band is within the "slow" or "fast" rotation range.
        /// </summary>
        private void ApplyYarnFreqCorrection(float[] freqs, int halfDim)
        {
            float lowFreqWavelen = (float)(_ropeOrigCtx / _ropeBetaSlow);
            float highFreqWavelen = (float)(_ropeOrigCtx / _ropeBetaFast);

            for (int i = 0; i < halfDim; i++)
            {
                float origFreq = 1.0f / MathF.Pow(Config.RopeBase, (2.0f * i) / _ropeDim);
                float wavelen = 2.0f * MathF.PI / origFreq;

                if (wavelen < highFreqWavelen)
                {
                    freqs[i] = origFreq;   // high frequency: extrapolation
                }
                else if (wavelen > lowFreqWavelen)
                {
                    freqs[i] = origFreq / Config.RopeScale;   // low frequency: interpolation
                }
                else
                {
                    float smooth = (lowFreqWavelen / wavelen - 1.0f) /
                                   (lowFreqWavelen / highFreqWavelen - 1.0f);
                    float interpFreq = origFreq / Config.RopeScale;
                    freqs[i] = (1.0f - smooth) * interpFreq + smooth * origFreq;
                }
            }
        }

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

        /// <summary>GPT-J (norm) RoPE with Mistral's YaRN parameters, then the YaRN
        /// position-dependent Q scale. A single decode token uses the precomputed
        /// YaRN-corrected frequencies on the host; prompts and batches use Ops.RoPEEx.</summary>
        protected override void RotateQueryKey(RankScope rank, int layer, ref Tensor q, ref Tensor k, TokenPositions positions)
        {
            if (positions.Count == 1 && positions.IsContiguous)
            {
                ApplyRoPEDecode(q, rank.LocalHeads, positions.Start);
                ApplyRoPEDecode(k, rank.LocalKvHeads, positions.Start);
            }
            else
            {
                q = ApplyRoPE(q, rank.LocalHeads, positions);
                k = ApplyRoPE(k, rank.LocalKvHeads, positions);
            }
            if (_ropeOrigCtx > 0)
                ApplyPositionScale(q, rank.LocalHeads * KeyDim, positions);
        }

        private unsafe void ApplyRoPEDecode(Tensor data, int numHeads, int position)
        {
            int halfDim = _ropeDim / 2;
            float* ptr = GetFloatPtr(data);
            float* cosTable = stackalloc float[halfDim];
            float* sinTable = stackalloc float[halfDim];
            for (int i = 0; i < halfDim; i++)
            {
                float theta = position * _ropeFreqs[i];
                cosTable[i] = MathF.Cos(theta);
                sinTable[i] = MathF.Sin(theta);
            }
            for (int h = 0; h < numHeads; h++)
            {
                float* head = ptr + h * KeyDim;
                for (int i = 0; i < halfDim; i++)
                {
                    float x0 = head[2 * i];
                    float x1 = head[2 * i + 1];
                    head[2 * i] = x0 * cosTable[i] - x1 * sinTable[i];
                    head[2 * i + 1] = x0 * sinTable[i] + x1 * cosTable[i];
                }
            }
            HostWritten(data);
        }

        private Tensor ApplyRoPE(Tensor data, int numHeads, TokenPositions positions)
        {
            int tokens = positions.Count;
            using Tensor posTensor = CreateIntTensorOn(data.Storage.Allocator, positions.PerHead(numHeads), tokens * numHeads);
            HostWritten(posTensor);
            // In place: rotating into a fresh tensor and disposing the input while the device
            // op is still queued (Metal lazy-sync) gave a deterministic, wrong prefill.
            using Tensor reshaped = data.View(1, tokens, numHeads, KeyDim);
            Ops.RoPEEx(
                reshaped, reshaped, posTensor, _ropeDim, 0, _ropeOrigCtx,
                Config.RopeBase, 1.0f / Config.RopeScale,
                _ropeType == "yarn" ? _ropeExtFactor : 0f,
                ComputeAttnFactor(),
                _ropeType == "yarn" ? _ropeBetaFast : 0f,
                _ropeType == "yarn" ? _ropeBetaSlow : 0f);
            return data;
        }

        private float ComputeAttnFactor()
        {
            if (_ropeMscale != 0 && _ropeMscaleAllDim != 0)
                return 1.0f / (0.1f * MathF.Log(Config.RopeScale) + 1.0f);
            return 1.0f;
        }

        /// <summary>Position-dependent Q scaling for YaRN, one factor per token (it scales the
        /// whole row): q *= (1 + beta * log(1 + floor(pos / orig_ctx))).</summary>
        private unsafe void ApplyPositionScale(Tensor q, int rowWidth, TokenPositions positions)
        {
            float* ptr = GetFloatPtr(q);
            for (int t = 0; t < positions.Count; t++)
            {
                float interval = MathF.Floor((float)positions[t] / _ropeOrigCtx);
                float posScale = 1.0f + _ropeScalingBeta * MathF.Log(1.0f + interval);
                if (MathF.Abs(posScale - 1.0f) < 1e-7f) continue;
                VecScale(ptr + (long)t * rowWidth, posScale, rowWidth);
            }
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
