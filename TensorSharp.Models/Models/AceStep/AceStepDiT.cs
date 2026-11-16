// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ACE-Step 1.5's AceStepConditionGenerationModel (ComfyUI comfy/ldm/ace/ace_step15.py):
//
//   condition encoder   lyric token embeddings -> 8 blocks; the timbre reference latent -> 4 blocks,
//                       first row; the caption's hidden states projected; joined [lyrics, timbre, caption]
//   audio-code hints    5 Hz codes -> FSQ (levels 8,8,8,5,5,5) -> project_out -> detokenizer (each code
//                       expanded to 5 frames with learned offsets, 2 blocks within each group) -> 25 Hz
//   DiT                 [hints, ones, noisy latent] patchified by 2 -> 24 blocks of AdaLN-modulated
//                       self-attention (sliding window 128 on even layers), cross-attention over the
//                       condition, and a gated SwiGLU MLP -> velocity [T, 64]
//
// The condition and the cross-attention keys/values are computed once per song; each Euler step
// runs only the DiT blocks.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Models.Direct;
using TensorSharp.Runtime;

namespace TensorSharp.Models.AceStep
{
    internal sealed class AceStepDiT : IDisposable
    {
        private const int Heads = 16, KvHeads = 8, HeadDim = 128, Hidden = 2048, Window = 128, Patch = 2, LatentDim = 64;
        private const float Theta = 1_000_000f;
        private static readonly int[] FsqLevels = { 8, 8, 8, 5, 5, 5 };

        private readonly DirectContext _ctx;
        private readonly GgufFile _gguf;

        // condition encoder
        private readonly DirectLinear _textProjector, _lyricIn, _timbreIn;
        private readonly AceEncoderLayer[] _lyricLayers, _timbreLayers;
        private readonly Tensor _lyricNorm, _timbreNorm;
        // audio-code hints
        private readonly DirectLinear _fsqOut, _detokIn, _detokOut;
        private readonly AceEncoderLayer[] _detokLayers;
        private readonly Tensor _detokNorm;
        private readonly float[] _detokSpecial;   // [5, 2048]
        // DiT
        private readonly DirectLinear _projIn, _projOut, _condEmbed;
        private readonly TimeEmbed _time, _timeR;
        private readonly Layer[] _layers;
        private readonly Tensor _normOut;
        private readonly float[] _outTable;        // [2, 2048]

        private sealed class TimeEmbed
        {
            public DirectLinear L1, L2, Proj;
        }

        private sealed class Layer
        {
            public Tensor SelfNorm, CrossNorm, MlpNorm;
            public AceAttention Self, Cross;
            public AceMlp Mlp;
            public float[] Table;                  // [6, 2048]
            public bool Sliding;
            public Tensor CrossK, CrossV;          // per song
        }

        public AceStepDiT(DirectContext ctx, GgufFile gguf)
        {
            _ctx = ctx;
            _gguf = gguf;
            const string e = "dit.encoder.";
            _textProjector = DirectLinear.FromGguf(ctx, gguf, e + "text_projector.weight");
            _lyricIn = DirectLinear.FromGguf(ctx, gguf, e + "lyric_encoder.embed_tokens.weight", e + "lyric_encoder.embed_tokens.bias");
            _lyricLayers = Blocks(e + "lyric_encoder.layers.");
            _lyricNorm = AceOps.Vec(ctx, gguf, e + "lyric_encoder.norm.weight");
            _timbreIn = DirectLinear.FromGguf(ctx, gguf, e + "timbre_encoder.embed_tokens.weight", e + "timbre_encoder.embed_tokens.bias");
            _timbreLayers = Blocks(e + "timbre_encoder.layers.");
            _timbreNorm = AceOps.Vec(ctx, gguf, e + "timbre_encoder.norm.weight");

            _fsqOut = DirectLinear.FromGguf(ctx, gguf, "dit.tokenizer.quantizer.project_out.weight", "dit.tokenizer.quantizer.project_out.bias");
            _detokIn = DirectLinear.FromGguf(ctx, gguf, "dit.detokenizer.embed_tokens.weight", "dit.detokenizer.embed_tokens.bias");
            _detokLayers = Blocks("dit.detokenizer.layers.");
            _detokNorm = AceOps.Vec(ctx, gguf, "dit.detokenizer.norm.weight");
            _detokOut = DirectLinear.FromGguf(ctx, gguf, "dit.detokenizer.proj_out.weight", "dit.detokenizer.proj_out.bias");
            _detokSpecial = DirectOps.DequantTensor(gguf, "dit.detokenizer.special_tokens");

            const string d = "dit.decoder.";
            _projIn = DirectLinear.FromGgufPatch(ctx, gguf, d + "proj_in.1.weight", d + "proj_in.1.bias");
            // proj_out is ConvTranspose1d(2048 -> 64, k 2, s 2): weight [in, out, k]; as a linear from a row of
            // 2048 to the (out, k) pairs it is the transpose of the stored [2048, 128] block.
            float[] wt = DirectOps.DequantTensor(gguf, d + "proj_out.1.weight");
            float[] bt = DirectOps.DequantTensor(gguf, d + "proj_out.1.bias");
            var w = new float[(long)LatentDim * Patch * Hidden];
            for (int c = 0; c < Hidden; c++)
                for (int ok = 0; ok < LatentDim * Patch; ok++)
                    w[(long)ok * Hidden + c] = wt[(long)c * LatentDim * Patch + ok];
            var b = new float[LatentDim * Patch];
            for (int ok = 0; ok < b.Length; ok++) b[ok] = bt[ok / Patch];
            _projOut = DirectLinear.FromFloats(ctx, w, Hidden, LatentDim * Patch, b);
            _condEmbed = DirectLinear.FromGguf(ctx, gguf, d + "condition_embedder.weight", d + "condition_embedder.bias");
            _time = LoadTime(d + "time_embed.");
            _timeR = LoadTime(d + "time_embed_r.");
            _normOut = AceOps.Vec(ctx, gguf, d + "norm_out.weight");
            _outTable = DirectOps.DequantTensor(gguf, d + "scale_shift_table");
            var layers = new List<Layer>();
            for (int i = 0; gguf.Tensors.ContainsKey($"{d}layers.{i}.self_attn_norm.weight"); i++)
            {
                string p = $"{d}layers.{i}.";
                layers.Add(new Layer
                {
                    SelfNorm = AceOps.Vec(ctx, gguf, p + "self_attn_norm.weight"),
                    CrossNorm = AceOps.Vec(ctx, gguf, p + "cross_attn_norm.weight"),
                    MlpNorm = AceOps.Vec(ctx, gguf, p + "mlp_norm.weight"),
                    Self = AceAttention.Load(ctx, gguf, p + "self_attn.", Heads, KvHeads, HeadDim),
                    Cross = AceAttention.Load(ctx, gguf, p + "cross_attn.", Heads, KvHeads, HeadDim),
                    Mlp = AceMlp.Load(ctx, gguf, p + "mlp."),
                    Table = DirectOps.DequantTensor(gguf, p + "scale_shift_table"),
                    Sliding = i % 2 == 0,
                });
            }
            _layers = layers.ToArray();
        }

        private AceEncoderLayer[] Blocks(string prefix)
        {
            var list = new List<AceEncoderLayer>();
            for (int i = 0; _gguf.Tensors.ContainsKey($"{prefix}{i}.input_layernorm.weight"); i++)
                list.Add(new AceEncoderLayer(_ctx, _gguf, $"{prefix}{i}.", Heads, KvHeads, HeadDim));
            return list.ToArray();
        }

        private TimeEmbed LoadTime(string p) => new()
        {
            L1 = DirectLinear.FromGguf(_ctx, _gguf, p + "linear_1.weight", p + "linear_1.bias"),
            L2 = DirectLinear.FromGguf(_ctx, _gguf, p + "linear_2.weight", p + "linear_2.bias"),
            Proj = DirectLinear.FromGguf(_ctx, _gguf, p + "time_proj.weight", p + "time_proj.bias"),
        };

        public int LayerCount => _layers.Length;

        // ------------------------------------------------------------------ per-song conditioning
        /// <summary>
        /// The condition [lyrics, timbre, caption] for cross-attention, bound into every layer's keys and
        /// values. <paramref name="caption"/> is the caption encoder's last hidden state [tc, 1024], <paramref
        /// name="lyrics"/> the lyric token embeddings [tl, 1024], <paramref name="timbre"/> the reference
        /// latent rows [tt, 64].
        /// </summary>
        public void BindCondition(float[] caption, int tc, float[] lyrics, int tl, float[] timbre, int tt)
        {
            using var lyricRows = _ctx.FromFloats(lyrics, tl, lyrics.Length / tl);
            using var lyric = _lyricIn.Forward(lyricRows);
            int[] lp = AceOps.Range(0, tl);
            foreach (var layer in _lyricLayers) layer.Forward(_ctx, lyric, lp, Theta, null);
            using var lyricOut = DirectOps.RmsNorm(_ctx, lyric, _lyricNorm, AceOps.Eps);

            using var timbreRows = _ctx.FromFloats(timbre, tt, LatentDim);
            using var tim = _timbreIn.Forward(timbreRows);
            int[] tp = AceOps.Range(0, tt);
            foreach (var layer in _timbreLayers) layer.Forward(_ctx, tim, tp, Theta, null);
            using var timOut = DirectOps.RmsNorm(_ctx, tim, _timbreNorm, AceOps.Eps);
            using var timFirst = timOut.Narrow(0, 0, 1);

            using var capRows = _ctx.FromFloats(caption, tc, caption.Length / tc);
            using var cap = _textProjector.Forward(capRows);

            int total = tl + 1 + tc;
            using var cond = _ctx.NewF32(total, Hidden);
            using (var dst = cond.Narrow(0, 0, tl)) Ops.Copy(dst, lyricOut);
            using (var dst = cond.Narrow(0, tl, 1)) Ops.Copy(dst, timFirst);
            using (var dst = cond.Narrow(0, tl + 1, tc)) Ops.Copy(dst, cap);
            using var enc = _condEmbed.Forward(cond);
            foreach (var layer in _layers)
            {
                layer.CrossK?.Dispose();
                layer.CrossV?.Dispose();
                using var k0 = layer.Cross.K.Forward(enc);
                layer.CrossK = AceOps.HeadNorm(_ctx, k0, layer.Cross.KNorm, KvHeads, HeadDim);
                layer.CrossV = layer.Cross.V.Forward(enc);
            }
        }

        /// <summary>The 25 Hz hints [frames, 64] the 5 Hz audio codes decode to (codes padded with 35847
        /// to cover the latent, as ComfyUI does).</summary>
        public float[] Hints(IReadOnlyList<int> codes, int frames)
        {
            var c = new List<int>(codes);
            int need = (frames + 4) / 5;
            while (c.Count < need) c.Add(35847);
            int n = c.Count;
            int[] basis = new int[FsqLevels.Length];
            basis[0] = 1;
            for (int i = 1; i < basis.Length; i++) basis[i] = basis[i - 1] * FsqLevels[i - 1];
            var fsq = new float[(long)n * FsqLevels.Length];
            for (int t = 0; t < n; t++)
                for (int j = 0; j < FsqLevels.Length; j++)
                {
                    int level = (c[t] / basis[j]) % FsqLevels[j];
                    fsq[(long)t * FsqLevels.Length + j] = level * (2f / (FsqLevels[j] - 1)) - 1f;
                }
            using var fsqT = _ctx.FromFloats(fsq, n, FsqLevels.Length);
            using var h5 = _fsqOut.Forward(fsqT);              // [n, 2048]
            using var e = _detokIn.Forward(h5);
            float[] eh = DirectOps.ToArray(e);
            const int p = 5;
            var x = new float[(long)n * p * Hidden];
            for (int t = 0; t < n; t++)
                for (int k = 0; k < p; k++)
                    for (int ch = 0; ch < Hidden; ch++)
                        x[((long)t * p + k) * Hidden + ch] = eh[(long)t * Hidden + ch] + _detokSpecial[(long)k * Hidden + ch];
            int rows = n * p;
            using var xt = _ctx.FromFloats(x, rows, Hidden);
            // Each code's 5 frames attend among themselves: a block-diagonal bias, positions 0..4 per block.
            var bias = new float[(long)Heads * rows * rows];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < rows; j++)
                    if (i / p != j / p)
                        for (int h = 0; h < Heads; h++) bias[((long)h * rows + i) * rows + j] = float.MinValue / 4;
            using var biasT = _ctx.FromFloats(bias, Heads, rows, rows);
            int[] pos = Enumerable.Range(0, rows).Select(i => i % p).ToArray();
            foreach (var layer in _detokLayers) layer.Forward(_ctx, xt, pos, Theta, biasT);
            using var normed = DirectOps.RmsNorm(_ctx, xt, _detokNorm, AceOps.Eps);
            using var outT = _detokOut.Forward(normed);
            float[] all = DirectOps.ToArray(outT);             // [rows, 64]
            return all.AsSpan(0, frames * LatentDim).ToArray();
        }

        // ------------------------------------------------------------------ one denoising step
        /// <summary>Velocity [frames, 64] for the noisy latent x [frames, 64] at time t, given the
        /// hints [frames, 64] (the context is [hints, ones]).</summary>
        public float[] Velocity(float[] x, float[] hints, int frames, float t)
        {
            int padded = frames + (frames % Patch);
            int tokens = padded / Patch;
            const int inCh = 3 * LatentDim;                     // context (hints, ones) then the latent
            var patches = new float[(long)tokens * inCh * Patch];
            for (int f = 0; f < frames; f++)
            {
                int p = f / Patch, k = f % Patch;
                long row = (long)p * inCh * Patch;
                for (int ch = 0; ch < LatentDim; ch++)
                {
                    patches[row + ch * Patch + k] = hints[(long)f * LatentDim + ch];
                    patches[row + (LatentDim + ch) * Patch + k] = 1f;
                    patches[row + (2 * LatentDim + ch) * Patch + k] = x[(long)f * LatentDim + ch];
                }
            }

            // Time embeddings: t, and t - t_r = 0.
            var (tembT, projT) = TimeEmbedding(_time, t);
            var (tembR, projR) = TimeEmbedding(_timeR, 0f);
            var temb = new float[Hidden];
            var proj = new float[6 * Hidden];
            for (int i = 0; i < Hidden; i++) temb[i] = tembT[i] + tembR[i];
            for (int i = 0; i < proj.Length; i++) proj[i] = projT[i] + projR[i];

            using var pt = _ctx.FromFloats(patches, tokens, inCh * Patch);
            using var h = _projIn.Forward(pt);                  // [tokens, 2048]
            int[] pos = AceOps.Range(0, tokens);
            using var window = AceOps.WindowBias(_ctx, Heads, tokens, Window);
            foreach (var layer in _layers)
            {
                var mod = new float[6 * Hidden];
                for (int i = 0; i < mod.Length; i++) mod[i] = layer.Table[i] + proj[i];
                using var shiftMsa = Row(mod, 0); using var scaleMsa = Row(mod, 1); using var gateMsa = Row(mod, 2);
                using var cShift = Row(mod, 3); using var cScale = Row(mod, 4); using var cGate = Row(mod, 5);

                using (var n1 = DirectOps.RmsNorm(_ctx, h, layer.SelfNorm, AceOps.Eps))
                {
                    DirectOps.ModulateRows(_ctx, n1, n1, shiftMsa, scaleMsa, 0, tokens);
                    using var a = AceOps.Attend(_ctx, layer.Self, n1, n1, pos, pos, Theta, layer.Sliding ? window : null);
                    DirectOps.GateAddRows(_ctx, h, a, gateMsa, 0, tokens);
                }
                using (var n2 = DirectOps.RmsNorm(_ctx, h, layer.CrossNorm, AceOps.Eps))
                {
                    using var q0 = layer.Cross.Q.Forward(n2);
                    using var q = AceOps.HeadNorm(_ctx, q0, layer.Cross.QNorm, Heads, HeadDim);
                    using var kf = AceOps.ExpandKv(_ctx, layer.CrossK, Heads, KvHeads, HeadDim);
                    using var vf = AceOps.ExpandKv(_ctx, layer.CrossV, Heads, KvHeads, HeadDim);
                    using var o = DirectOps.Attention(_ctx, q, kf, vf, Heads, HeadDim, 1f / MathF.Sqrt(HeadDim));
                    using var ao = layer.Cross.O.Forward(o);
                    Ops.Add(h, h, ao);
                }
                using (var n3 = DirectOps.RmsNorm(_ctx, h, layer.MlpNorm, AceOps.Eps))
                {
                    DirectOps.ModulateRows(_ctx, n3, n3, cShift, cScale, 0, tokens);
                    using var m = layer.Mlp.Forward(_ctx, n3);
                    DirectOps.GateAddRows(_ctx, h, m, cGate, 0, tokens);
                }
            }
            var outMod = new float[2 * Hidden];
            for (int i = 0; i < Hidden; i++)
            {
                outMod[i] = _outTable[i] + temb[i];
                outMod[Hidden + i] = _outTable[Hidden + i] + temb[i];
            }
            using var shift = Row(outMod, 0);
            using var scale = Row(outMod, 1);
            using var hn = DirectOps.RmsNorm(_ctx, h, _normOut, AceOps.Eps);
            DirectOps.ModulateRows(_ctx, hn, hn, shift, scale, 0, tokens);
            using var y = _projOut.Forward(hn);                 // [tokens, 64 * 2] as (out, k)
            float[] yh = DirectOps.ToArray(y);
            var v = new float[(long)frames * LatentDim];
            for (int f = 0; f < frames; f++)
            {
                int p = f / Patch, k = f % Patch;
                for (int ch = 0; ch < LatentDim; ch++)
                    v[(long)f * LatentDim + ch] = yh[(long)p * LatentDim * Patch + ch * Patch + k];
            }
            return v;
        }

        private Tensor Row(float[] data, int index) =>
            _ctx.FromFloats(data.AsSpan(index * Hidden, Hidden).ToArray(), Hidden);

        /// <summary>TimestepEmbedding: sinusoid(1000 t, 256; cos then sin) -> linear, SiLU, linear -> temb;
        /// time_proj(SiLU(temb)) -> 6 modulation rows.</summary>
        private (float[] Temb, float[] Proj) TimeEmbedding(TimeEmbed te, float t)
        {
            const int dim = 256, half = dim / 2;
            var freq = new float[dim];
            float tt = 1000f * t;
            for (int i = 0; i < half; i++)
            {
                float f = MathF.Exp(-MathF.Log(10000f) * i / half);
                float a = tt * f;
                freq[i] = MathF.Cos(a);
                freq[half + i] = MathF.Sin(a);
            }
            using var ft = _ctx.FromFloats(freq, 1, dim);
            using var h1 = te.L1.Forward(ft);
            using var s1 = DirectOps.Silu(_ctx, h1);
            using var temb = te.L2.Forward(s1);
            using var s2 = DirectOps.Silu(_ctx, temb);
            using var proj = te.Proj.Forward(s2);
            return (DirectOps.ToArray(temb), DirectOps.ToArray(proj));
        }

        public void Dispose()
        {
            foreach (var l in _layers)
            {
                l.CrossK?.Dispose();
                l.CrossV?.Dispose();
            }
        }
    }
}
