// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Whisper's encoder and decoder on the direct tensor primitives (DirectOps), so the same code runs
// on --backend cpu (managed BLAS) and --backend cuda (resident F16 weights through cuBLAS).
//
//   encoder: conv1 (k3) -> GELU -> conv2 (k3, stride 2) -> GELU -> + sinusoidal positions
//            -> N pre-norm blocks (self-attention, GELU MLP) -> LayerNorm
//   decoder: token + learned positions -> N blocks (causal self-attention with a KV cache,
//            cross-attention over the encoder output, GELU MLP) -> LayerNorm -> tied logits
//
// The two convolutions run as im2col matmuls on the host-side column matrix; everything after
// them stays on the device.
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using TensorSharp.Models.Direct;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Whisper
{
    internal sealed class WhisperNetwork : IDisposable
    {
        private const float LayerNormEps = 1e-5f;

        private sealed class Attn
        {
            public DirectLinear Q, K, V, O;
        }

        private sealed class EncoderLayer
        {
            public Tensor Ln1G, Ln1B, Ln2G, Ln2B;
            public Attn Self;
            public DirectLinear Fc1, Fc2;
        }

        private sealed class DecoderLayer
        {
            public Tensor LnSelfG, LnSelfB, LnCrossG, LnCrossB, LnFfnG, LnFfnB;
            public Attn Self, Cross;
            public DirectLinear Fc1, Fc2;
            public Tensor KCache, VCache;   // [maxTarget, d]
            public Tensor CrossK, CrossV;   // [frames, d] for the current window
            public float[] CrossKHost;      // kept for the alignment heads' attention weights
        }

        private readonly DirectContext _ctx;
        private readonly GgufFile _gguf;
        private readonly WhisperConfig _cfg;
        private readonly DirectLinear _conv1, _conv2, _logits;
        private readonly Tensor _encPos, _encLnG, _encLnB, _decLnG, _decLnB;
        private readonly float[] _decPos;     // [maxTarget, d]
        private readonly EncoderLayer[] _enc;
        private readonly DecoderLayer[] _dec;
        private readonly int _d, _encHeads, _decHeads, _headDim;
        private readonly HashSet<int> _alignmentLayers = new();
        private int _crossFrames;

        public WhisperNetwork(DirectContext ctx, GgufFile gguf, WhisperConfig cfg)
        {
            _ctx = ctx;
            _gguf = gguf;
            _cfg = cfg;
            _d = cfg.DModel;
            _encHeads = cfg.EncoderHeads;
            _decHeads = cfg.DecoderHeads;
            _headDim = _d / _encHeads;
            if (_d / _decHeads != _headDim)
                throw new NotSupportedException("encoder and decoder heads of different widths are not supported.");
            foreach (var (layer, _) in cfg.AlignmentHeads) _alignmentLayers.Add(layer);

            _conv1 = DirectLinear.FromGgufPatch(ctx, gguf, "model.encoder.conv1.weight", "model.encoder.conv1.bias");
            _conv2 = DirectLinear.FromGgufPatch(ctx, gguf, "model.encoder.conv2.weight", "model.encoder.conv2.bias");
            _encPos = Vec("model.encoder.embed_positions.weight");
            _encLnG = Vec("model.encoder.layer_norm.weight");
            _encLnB = Vec("model.encoder.layer_norm.bias");
            _decLnG = Vec("model.decoder.layer_norm.weight");
            _decLnB = Vec("model.decoder.layer_norm.bias");
            _decPos = DirectOps.DequantTensor(gguf, "model.decoder.embed_positions.weight");
            _logits = DirectLinear.FromGguf(ctx, gguf, "model.decoder.embed_tokens.weight");

            _enc = new EncoderLayer[cfg.EncoderLayers];
            for (int i = 0; i < _enc.Length; i++)
            {
                string p = $"model.encoder.layers.{i}.";
                _enc[i] = new EncoderLayer
                {
                    Ln1G = Vec(p + "self_attn_layer_norm.weight"), Ln1B = Vec(p + "self_attn_layer_norm.bias"),
                    Ln2G = Vec(p + "final_layer_norm.weight"), Ln2B = Vec(p + "final_layer_norm.bias"),
                    Self = AttnOf(p + "self_attn."),
                    Fc1 = DirectLinear.FromGguf(ctx, gguf, p + "fc1.weight", p + "fc1.bias"),
                    Fc2 = DirectLinear.FromGguf(ctx, gguf, p + "fc2.weight", p + "fc2.bias"),
                };
            }
            _dec = new DecoderLayer[cfg.DecoderLayers];
            for (int i = 0; i < _dec.Length; i++)
            {
                string p = $"model.decoder.layers.{i}.";
                _dec[i] = new DecoderLayer
                {
                    LnSelfG = Vec(p + "self_attn_layer_norm.weight"), LnSelfB = Vec(p + "self_attn_layer_norm.bias"),
                    LnCrossG = Vec(p + "encoder_attn_layer_norm.weight"), LnCrossB = Vec(p + "encoder_attn_layer_norm.bias"),
                    LnFfnG = Vec(p + "final_layer_norm.weight"), LnFfnB = Vec(p + "final_layer_norm.bias"),
                    Self = AttnOf(p + "self_attn."),
                    Cross = AttnOf(p + "encoder_attn."),
                    Fc1 = DirectLinear.FromGguf(ctx, gguf, p + "fc1.weight", p + "fc1.bias"),
                    Fc2 = DirectLinear.FromGguf(ctx, gguf, p + "fc2.weight", p + "fc2.bias"),
                    KCache = ctx.Own(ctx.NewF32(cfg.MaxTargetPositions, _d)),
                    VCache = ctx.Own(ctx.NewF32(cfg.MaxTargetPositions, _d)),
                };
            }
        }

        private Tensor Vec(string name)
        {
            float[] v = DirectOps.DequantTensor(_gguf, name);
            var info = _gguf.Tensors[name];
            if (info.Shape.Length == 2)
                return _ctx.Own(_ctx.FromFloats(v, (long)info.Shape[1], (long)info.Shape[0]));
            return _ctx.Own(_ctx.FromFloats(v, v.LongLength));
        }

        private Attn AttnOf(string p) => new()
        {
            Q = DirectLinear.FromGguf(_ctx, _gguf, p + "q_proj.weight", p + "q_proj.bias"),
            K = DirectLinear.FromGguf(_ctx, _gguf, p + "k_proj.weight", p + "k_proj.bias"),
            V = DirectLinear.FromGguf(_ctx, _gguf, p + "v_proj.weight", p + "v_proj.bias"),
            O = DirectLinear.FromGguf(_ctx, _gguf, p + "out_proj.weight", p + "out_proj.bias"),
        };

        private float Scale => 1f / MathF.Sqrt(_headDim);

        /// <summary>Encode one window of log-mel features [nMels, chunkFrames] and bind the decoder's
        /// cross-attention keys and values to it.</summary>
        public void Encode(float[] mel, bool keepAlignmentKeys)
        {
            int nMels = _cfg.NumMelBins, frames = _cfg.ChunkFrames;
            int d = _d;
            if (mel.Length != (long)nMels * frames)
                throw new ArgumentException($"expected {nMels} x {frames} mel features, got {mel.Length} values.");

            // conv1: kernel 3, padding 1. Column rows hold (channel, tap) with the tap fastest,
            // the order of the flattened [out, in, k] kernel.
            var col1 = new float[(long)frames * nMels * 3];
            Parallel.For(0, frames, t =>
            {
                long row = (long)t * nMels * 3;
                for (int c = 0; c < nMels; c++)
                    for (int k = 0; k < 3; k++)
                    {
                        int s = t + k - 1;
                        col1[row + c * 3 + k] = s >= 0 && s < frames ? mel[(long)c * frames + s] : 0f;
                    }
            });
            float[] h1;
            using (var x1 = _ctx.FromFloats(col1, frames, (long)nMels * 3))
            using (var y1 = _conv1.Forward(x1))
            using (var g1 = DirectOps.GeluErf(_ctx, y1))
                h1 = DirectOps.ToArray(g1);               // [frames, d]

            // conv2: kernel 3, stride 2, padding 1.
            int positions = frames / 2;
            var col2 = new float[(long)positions * d * 3];
            Parallel.For(0, positions, t =>
            {
                long row = (long)t * d * 3;
                for (int k = 0; k < 3; k++)
                {
                    int s = 2 * t + k - 1;
                    bool inside = s >= 0 && s < frames;
                    for (int c = 0; c < d; c++)
                        col2[row + c * 3 + k] = inside ? h1[(long)s * d + c] : 0f;
                }
            });
            Tensor x;
            using (var x2 = _ctx.FromFloats(col2, positions, (long)d * 3))
            using (var y2 = _conv2.Forward(x2))
                x = DirectOps.GeluErf(_ctx, y2);             // [positions, d]
            using (var pos = _encPos.Narrow(0, 0, positions))
                Ops.Add(x, x, pos);

            foreach (var layer in _enc)
            {
                using (var xn = DirectOps.LayerNorm(_ctx, x, layer.Ln1G, layer.Ln1B, LayerNormEps))
                using (var q = layer.Self.Q.Forward(xn))
                using (var k = layer.Self.K.Forward(xn))
                using (var v = layer.Self.V.Forward(xn))
                using (var a = DirectOps.Attention(_ctx, q, k, v, _encHeads, _headDim, Scale))
                using (var o = layer.Self.O.Forward(a))
                    Ops.Add(x, x, o);
                using (var xn = DirectOps.LayerNorm(_ctx, x, layer.Ln2G, layer.Ln2B, LayerNormEps))
                using (var h = layer.Fc1.Forward(xn))
                using (var g = DirectOps.GeluErf(_ctx, h))
                using (var y = layer.Fc2.Forward(g))
                    Ops.Add(x, x, y);
            }
            using (x)
            using (var enc = DirectOps.LayerNorm(_ctx, x, _encLnG, _encLnB, LayerNormEps))
            {
                for (int i = 0; i < _dec.Length; i++)
                {
                    var layer = _dec[i];
                    layer.CrossK?.Dispose();
                    layer.CrossV?.Dispose();
                    layer.CrossK = layer.Cross.K.Forward(enc);
                    layer.CrossV = layer.Cross.V.Forward(enc);
                    layer.CrossKHost = keepAlignmentKeys && _alignmentLayers.Contains(i) ? DirectOps.ToArray(layer.CrossK) : null;
                }
            }
            _crossFrames = positions;
        }

        /// <summary>What a decoder pass hands back: logits of the rows asked for and, when asked,
        /// the cross-attention queries of the alignment layers for every row.</summary>
        public sealed class DecoderOutput
        {
            public float[][] Logits;
            public Dictionary<int, float[]> CrossQueries;
        }

        /// <summary>
        /// Run <paramref name="tokens"/> at positions start .. start+n-1 (the cache holds every
        /// earlier position), returning the vocabulary logits of <paramref name="logitRows"/>
        /// (indices into <paramref name="tokens"/>).
        /// </summary>
        public DecoderOutput Decode(IReadOnlyList<int> tokens, int start, IReadOnlyList<int> logitRows, bool captureCross = false)
        {
            int n = tokens.Count, d = _d;
            if (start + n > _cfg.MaxTargetPositions)
                throw new InvalidOperationException($"the decoder holds {_cfg.MaxTargetPositions} positions; {start + n} asked.");
            var ids = new int[n];
            for (int i = 0; i < n; i++) ids[i] = tokens[i];
            var emb = new float[(long)n * d];
            DirectOps.DequantRows(_gguf, "model.decoder.embed_tokens.weight", ids, emb);
            for (int i = 0; i < n; i++)
                for (int c = 0; c < d; c++)
                    emb[(long)i * d + c] += _decPos[(long)(start + i) * d + c];
            var x = _ctx.FromFloats(emb, n, d);

            Tensor causal = null;
            if (n > 1)
            {
                int sk = start + n;
                var bias = new float[(long)_decHeads * n * sk];
                for (int h = 0; h < _decHeads; h++)
                    for (int i = 0; i < n; i++)
                        for (int j = start + i + 1; j < sk; j++)
                            bias[((long)h * n + i) * sk + j] = -1e30f;
                causal = _ctx.FromFloats(bias, _decHeads, n, sk);
            }

            var output = new DecoderOutput { CrossQueries = captureCross ? new Dictionary<int, float[]>() : null };
            try
            {
                for (int li = 0; li < _dec.Length; li++)
                {
                    var layer = _dec[li];
                    using (var xn = DirectOps.LayerNorm(_ctx, x, layer.LnSelfG, layer.LnSelfB, LayerNormEps))
                    using (var q = layer.Self.Q.Forward(xn))
                    using (var k = layer.Self.K.Forward(xn))
                    using (var v = layer.Self.V.Forward(xn))
                    {
                        using (var kDst = layer.KCache.Narrow(0, start, n)) Ops.Copy(kDst, k);
                        using (var vDst = layer.VCache.Narrow(0, start, n)) Ops.Copy(vDst, v);
                        using var kAll = layer.KCache.Narrow(0, 0, start + n);
                        using var vAll = layer.VCache.Narrow(0, 0, start + n);
                        using var a = DirectOps.Attention(_ctx, q, kAll, vAll, _decHeads, _headDim, Scale, causal);
                        using var o = layer.Self.O.Forward(a);
                        Ops.Add(x, x, o);
                    }
                    using (var xn = DirectOps.LayerNorm(_ctx, x, layer.LnCrossG, layer.LnCrossB, LayerNormEps))
                    using (var q = layer.Cross.Q.Forward(xn))
                    {
                        if (captureCross && layer.CrossKHost != null)
                            output.CrossQueries[li] = DirectOps.ToArray(q);
                        using var a = DirectOps.Attention(_ctx, q, layer.CrossK, layer.CrossV, _decHeads, _headDim, Scale);
                        using var o = layer.Cross.O.Forward(a);
                        Ops.Add(x, x, o);
                    }
                    using (var xn = DirectOps.LayerNorm(_ctx, x, layer.LnFfnG, layer.LnFfnB, LayerNormEps))
                    using (var h = layer.Fc1.Forward(xn))
                    using (var g = DirectOps.GeluErf(_ctx, h))
                    using (var y = layer.Fc2.Forward(g))
                        Ops.Add(x, x, y);
                }

                output.Logits = new float[logitRows.Count][];
                if (logitRows.Count > 0)
                {
                    using var final = DirectOps.LayerNorm(_ctx, x, _decLnG, _decLnB, LayerNormEps);
                    for (int r = 0; r < logitRows.Count; r++)
                    {
                        using var row = final.Narrow(0, logitRows[r], 1);
                        using var logits = _logits.Forward(row);
                        output.Logits[r] = DirectOps.ToArray(logits);
                    }
                }
                return output;
            }
            finally
            {
                x.Dispose();
                causal?.Dispose();
            }
        }

        /// <summary>
        /// Raw scaled cross-attention scores q.k / sqrt(d) of one alignment head for every captured
        /// query row: [rows, frames] (openai/whisper's QK hooks, before the softmax).
        /// </summary>
        public float[] AlignmentScores(DecoderOutput output, int layer, int head, int rows, out int frames)
        {
            frames = _crossFrames;
            float[] q = output.CrossQueries[layer];
            float[] k = _dec[layer].CrossKHost;
            int d = _d, hd = _headDim, off = head * hd, nFrames = _crossFrames;
            float scale = Scale;
            var scores = new float[(long)rows * nFrames];
            Parallel.For(0, rows, r =>
            {
                var qr = new ReadOnlySpan<float>(q, r * d + off, hd);
                for (int f = 0; f < nFrames; f++)
                    scores[(long)r * nFrames + f] = scale *
                        System.Numerics.Tensors.TensorPrimitives.Dot(qr, new ReadOnlySpan<float>(k, f * d + off, hd));
            });
            return scores;
        }

        public void Dispose()
        {
            foreach (var layer in _dec)
            {
                layer.CrossK?.Dispose();
                layer.CrossV?.Dispose();
            }
        }
    }
}
