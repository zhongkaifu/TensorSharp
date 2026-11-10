// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Models
{
    public abstract partial class StandardDecoderModel
    {
        /// <summary>
        /// Rotary position embedding in <c>ggml_rope_ext</c>'s terms. Every path the runtime
        /// rotates through - <c>Ops.RoPEEx</c> on any backend, and the host decode rotation
        /// below - computes the same function from it, so a prompt's keys and a decoded query
        /// always agree.
        /// </summary>
        protected sealed class RotarySpec
        {
            private readonly float[] _invFreq;
            private readonly float[] _rampMix;

            /// <param name="dims">Rotated dimensions of a head (the rest pass through).</param>
            /// <param name="neoX">NeoX layout (pairs <c>i, i + dims/2</c>) rather than GPT-J (<c>2i, 2i+1</c>).</param>
            /// <param name="freqScale">1 / the context scaling factor.</param>
            /// <param name="extFactor">YaRN extrapolation mix; 0 disables YaRN.</param>
            /// <param name="attnFactor">Magnitude scale, as llama.cpp's <c>yarn_attn_factor</c>.</param>
            public RotarySpec(int dims, bool neoX, float freqBase, float freqScale = 1f, int originalContext = 0,
                float extFactor = 0f, float attnFactor = 1f, float betaFast = 0f, float betaSlow = 0f)
            {
                if (dims <= 0 || (dims & 1) != 0)
                    throw new ArgumentOutOfRangeException(nameof(dims), dims, "Rotary dims must be positive and even.");
                Dims = dims;
                NeoX = neoX;
                FreqBase = freqBase;
                FreqScale = freqScale;
                OriginalContext = originalContext;
                ExtFactor = extFactor;
                AttnFactor = attnFactor;
                BetaFast = betaFast;
                BetaSlow = betaSlow;

                int pairs = dims / 2;
                _invFreq = new float[pairs];
                _rampMix = new float[pairs];
                for (int i = 0; i < pairs; i++)
                    _invFreq[i] = MathF.Pow(freqBase, -2.0f * i / dims);

                // ggml_rope_yarn_corr_dims / rope_yarn_ramp / rope_yarn (ggml-cpu ops.cpp).
                Magnitude = attnFactor;
                if (extFactor != 0f)
                {
                    float low = MathF.Max(0f, MathF.Floor(CorrDim(dims, originalContext, betaFast, freqBase)));
                    float high = MathF.Min(dims - 1, MathF.Ceiling(CorrDim(dims, originalContext, betaSlow, freqBase)));
                    for (int i = 0; i < pairs; i++)
                    {
                        float y = (i - low) / MathF.Max(0.001f, high - low);
                        _rampMix[i] = (1f - MathF.Min(1f, MathF.Max(0f, y))) * extFactor;
                    }
                    Magnitude *= 1.0f + 0.1f * MathF.Log(1.0f / freqScale);
                }
            }

            public int Dims { get; }
            public bool NeoX { get; }
            public float FreqBase { get; }
            public float FreqScale { get; }
            public int OriginalContext { get; }
            public float ExtFactor { get; }
            public float AttnFactor { get; }
            public float BetaFast { get; }
            public float BetaSlow { get; }

            /// <summary>The factor cos/sin carry: the attention factor, times YaRN's
            /// <c>1 + 0.1 ln(1/freqScale)</c> when YaRN is on.</summary>
            public float Magnitude { get; }

            /// <summary>One position's cos/sin per rotated pair.</summary>
            internal void Table(int position, Span<float> cos, Span<float> sin)
            {
                bool yarn = ExtFactor != 0f;
                for (int i = 0; i < _invFreq.Length; i++)
                {
                    float extrap = position * _invFreq[i];
                    float interp = FreqScale * extrap;
                    float theta = yarn ? interp * (1f - _rampMix[i]) + extrap * _rampMix[i] : interp;
                    cos[i] = MathF.Cos(theta) * Magnitude;
                    sin[i] = MathF.Sin(theta) * Magnitude;
                }
            }

            private static float CorrDim(int dims, int originalContext, float rotations, float freqBase)
                => dims * MathF.Log(originalContext / (rotations * 2f * MathF.PI)) / (2f * MathF.Log(freqBase));
        }

        /// <summary>
        /// Whether a single sequence's decode token does its small per-head work (rotary, QK
        /// norm) on the host. Per-op device dispatch costs more than that arithmetic on the GGML
        /// backends and is free on the managed CPU; MLX and direct CUDA keep it on the device.
        /// Measured on an M5 Pro (Q4_K_M, interleaved runs), <c>ggml_metal</c> decode with the host
        /// rotary vs a device one: Ministral 3B 14.1 vs 11.9 tok/s, Hy-MT2 1.8B 14.8 vs 9.9.
        /// On MLX the device rotary is faster (Ministral 3B 50.9 vs 47.0).
        /// </summary>
        protected bool HostDecodeOps => _backend != BackendType.Mlx && _backend != BackendType.Cuda;

        /// <summary>
        /// Rotate <paramref name="data"/> ([tokens, heads * headDim], in place). A contiguous
        /// single-token decode on a <see cref="HostDecodeOps"/> backend rotates on the host;
        /// everything else (prompts, batches, MLX/CUDA) runs <c>Ops.RoPEEx</c>.
        /// </summary>
        protected void Rotate(Tensor data, int numHeads, int headDim, TokenPositions positions, RotarySpec spec)
        {
            if (positions.Count == 1 && positions.IsContiguous && HostDecodeOps)
            {
                RotateOnHost(data, numHeads, headDim, positions.Start, spec);
                HostWritten(data);
                return;
            }

            int tokens = positions.Count;
            using Tensor posTensor = CreateIntTensorOn(data.Storage.Allocator, positions.PerHead(numHeads), tokens * numHeads);
            HostWritten(posTensor);
            // In place: rotating into a fresh tensor and disposing the input while the device
            // op is still queued (Metal lazy-sync) gave a deterministic, wrong prefill.
            using Tensor reshaped = data.View(1, tokens, numHeads, headDim);
            Ops.RoPEEx(reshaped, reshaped, posTensor, spec.Dims, spec.NeoX ? 2 : 0, spec.OriginalContext,
                spec.FreqBase, spec.FreqScale, spec.ExtFactor, spec.AttnFactor, spec.BetaFast, spec.BetaSlow);
        }

        private unsafe void RotateOnHost(Tensor data, int numHeads, int headDim, int position, RotarySpec spec)
        {
            int pairs = Math.Min(spec.Dims, headDim) / 2;
            Span<float> cos = stackalloc float[spec.Dims / 2];
            Span<float> sin = stackalloc float[spec.Dims / 2];
            spec.Table(position, cos, sin);
            float* ptr = GetFloatPtr(data);
            for (int h = 0; h < numHeads; h++)
            {
                float* head = ptr + (long)h * headDim;
                if (spec.NeoX)
                {
                    int half = spec.Dims / 2;
                    for (int i = 0; i < pairs; i++)
                    {
                        float x0 = head[i];
                        float x1 = head[i + half];
                        head[i] = x0 * cos[i] - x1 * sin[i];
                        head[i + half] = x0 * sin[i] + x1 * cos[i];
                    }
                }
                else
                {
                    for (int i = 0; i < pairs; i++)
                    {
                        float x0 = head[2 * i];
                        float x1 = head[2 * i + 1];
                        head[2 * i] = x0 * cos[i] - x1 * sin[i];
                        head[2 * i + 1] = x0 * sin[i] + x1 * cos[i];
                    }
                }
            }
        }
    }
}
