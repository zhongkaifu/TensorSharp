// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Direct-backend MiniMax-H3 video VAE decoder: the ViT of
// TSGgml_MiniMaxH3VideoVaeDecode against the shared direct primitives, so
// BackendType.Cpu (the 100% pure-C# backend) can decode latents with no ggml.
//
// Three things differ from the DiT and are easy to get wrong:
//   - the QKV projection is PER-HEAD INTERLEAVED: head h owns
//     [h*3*hd, (h+1)*3*hd), not one contiguous q|k|v split;
//   - qk-norm here is RMSNorm with NO learnable gain;
//   - norm_out is a real LayerNorm (mean-centred, with bias), unlike the
//     RMSNorms inside the blocks.
using System;
using System.Collections.Generic;

using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models.Direct;
using TensorSharp.Runtime;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>Managed MiniMax-H3 video VAE decoder (no ggml).</summary>
    internal sealed class MiniMaxH3DirectVideoVae : IDisposable
    {
        private sealed class BlockW
        {
            public Tensor Norm1, Norm2, Scale1, Scale2;
            public DirectLinear Qkv, Out, W1, W2;
        }

        private readonly DirectContext _ctx;
        private readonly MiniMaxH3TensorOps _tensorOps;
        private readonly BlockW[] _blocks;
        private readonly DirectLinear _postQuant, _xEmbedder, _projOut;
        private readonly Tensor _registerTokens, _normOutW, _normOutB;
        private readonly List<DirectLinear> _owned = new();
        private readonly int _dim, _heads, _headDim, _inner, _rotDim, _numRegister;
        private readonly float _eps;
        private bool _disposed;

        /// <param name="read">Reads a VAE tensor by name as F32, or null when absent.</param>
        /// <param name="shape">Returns the [out, in] shape of a weight tensor.</param>
        public MiniMaxH3DirectVideoVae(
            IAllocator allocator,
            Func<string, float[]> read,
            Func<string, (long OutFeatures, long InFeatures)> shape,
            int numBlocks, int dim, int heads, int headDim, int inner, int rotDim,
            int numRegister, float eps,
            Func<string, (IntPtr Pointer, int Type, long Bytes)> mappedWeight = null)
        {
            _ctx = new DirectContext(allocator);
            _tensorOps = new MiniMaxH3TensorOps(_ctx);

            _dim = dim; _heads = heads; _headDim = headDim; _inner = inner;
            _rotDim = rotDim; _numRegister = numRegister; _eps = eps;

            DirectLinear Lin(string w, string b)
            {
                var (outF, inF) = shape(w);
                DirectLinear l;
                if (_ctx.IsCuda && mappedWeight != null &&
                    Environment.GetEnvironmentVariable("TS_H3_CUDA_MAPPED_VAE_WEIGHTS") == "1")
                {
                    var mapped = mappedWeight(w);
                    l = mapped.Type is 0 or 1
                        ? DirectLinear.FromMappedH3Dense(_ctx,mapped.Pointer,mapped.Type,inF,outF,mapped.Bytes,read(b))
                        : DirectLinear.FromFloats(_ctx,read(w),inF,outF,read(b));
                }
                else l = DirectLinear.FromFloats(_ctx, read(w), inF, outF, read(b), mlxDeviceOnly: _ctx.IsMlx);
                _owned.Add(l);
                return l;
            }
            Tensor Vec(string name)
            {
                float[] v = read(name);
                return v == null ? null : _ctx.Own(_ctx.FromFloats(v, v.LongLength));
            }

            _postQuant = Lin("post_quant_conv.weight", "post_quant_conv.bias");
            _xEmbedder = Lin("decoder.x_embedder.weight", "decoder.x_embedder.bias");
            _projOut = Lin("decoder.proj_out.weight", "decoder.proj_out.bias");
            _normOutW = Vec("decoder.norm_out.weight");
            _normOutB = Vec("decoder.norm_out.bias");

            float[] reg = read("decoder.register_tokens");
            _registerTokens = reg == null || numRegister <= 0
                ? null
                : _ctx.Own(_ctx.FromFloats(reg, numRegister, dim));

            _blocks = new BlockW[numBlocks];
            for (int i = 0; i < numBlocks; i++)
            {
                string p = "decoder.transformer_blocks." + i + ".";
                _blocks[i] = new BlockW
                {
                    Norm1 = Vec(p + "norm1.weight"),
                    Norm2 = Vec(p + "norm2.weight"),
                    Scale1 = Vec(p + "scale1"),
                    Scale2 = Vec(p + "scale2"),
                    Qkv = Lin(p + "attn.to_qkv.weight", p + "attn.to_qkv.bias"),
                    Out = Lin(p + "attn.to_out.weight", p + "attn.to_out.bias"),
                    W1 = Lin(p + "ff.w1.weight", p + "ff.w1.bias"),
                    W2 = Lin(p + "ff.w2.weight", p + "ff.w2.bias"),
                };
            }
        }

        /// <summary>
        /// Pull one of q/k/v out of the per-head interleaved QKV projection into a
        /// plain [seq, heads*hd] tensor.
        /// </summary>
        private unsafe Tensor SplitInterleaved(Tensor qkv, int seq, int which)
        {
            int heads = _heads, hd = _headDim;
            if (_tensorOps.IsAccelerated)
                return MiniMaxH3TensorOps.SplitInterleaved(qkv, seq, heads, hd, which);
            Tensor outT = _ctx.NewF32(seq, (long)heads * hd);
            float* src = (float*)CpuNativeHelpers.GetBufferStart(qkv);
            float* dst = (float*)CpuNativeHelpers.GetBufferStart(outT);
            long srcRow = (long)heads * 3 * hd, dstRow = (long)heads * hd;
            CpuWorkerPool.Shared.For(seq, t =>
            {
                float* s = src + t * srcRow;
                float* o = dst + t * dstRow;
                for (int h = 0; h < heads; h++)
                {
                    float* sh = s + (long)h * 3 * hd + (long)which * hd;
                    float* oh = o + (long)h * hd;
                    for (int c = 0; c < hd; c++) oh[c] = sh[c];
                }
            });
            return outT;
        }

        /// <summary>RMSNorm per head over the head dim, with no learnable gain.</summary>
        private Tensor NormHeadsNoGain(Tensor x, int seq)
        {
            using (x)
            using (Tensor flat = x.View((long)seq * _heads, _headDim))
            using (Tensor n = DirectOps.RmsNorm(_ctx, flat, _ctx.Ones(_headDim), _eps))
                return n.View(seq, (long)_heads * _headDim);
        }

        private unsafe void RopeHalf(Tensor x, float[] cos, float[] sin,
                                            int seq, int heads, int hd, int rot)
        {
            if (rot <= 0) return;
            if (_tensorOps.IsAccelerated)
            {
                _tensorOps.RopeHalf(x, cos, sin, seq, heads, hd, rot);
                return;
            }
            int half = rot / 2;
            float* px = (float*)CpuNativeHelpers.GetBufferStart(x);
            fixed (float* pcos = cos, psin = sin)
            {
                float* pc = pcos, ps = psin;
                CpuWorkerPool.Shared.For(seq, t =>
                {
                    float* ct = pc + (long)t * rot;
                    float* st = ps + (long)t * rot;
                    float* row = px + (long)t * heads * hd;
                    for (int h = 0; h < heads; h++)
                    {
                        float* v = row + (long)h * hd;
                        for (int i = 0; i < half; i++)
                        {
                            float a = v[i], b = v[i + half];
                            v[i] = a * ct[i] - b * st[i];
                            v[i + half] = b * ct[i + half] + a * st[i + half];
                        }
                    }
                });
            }
        }

        /// <summary>
        /// Decode one latent chunk. latent is [tokens, latentChannels]; the result
        /// is the raw patch tokens [tokens, patchDim].
        /// </summary>
        public float[] DecodeTokens(float[] latent, int tokens, int latentChannels,
                                    float[] cos, float[] sin, int patchDim)
        {
            // The reference appends the learned register tokens and ONE zero token
            // (torch.zeros_like, not the checkpoint mask_token, which is unused here).
            int seq = tokens + _numRegister + 1;

            Tensor h;
            using (Tensor lat = _ctx.FromFloats(latent, tokens, latentChannels))
            using (Tensor pq = _postQuant.Forward(lat))          // 1x1x1 Conv3d = per-token linear
                h = _xEmbedder.Forward(pq);                      // [tokens, dim]

            {
                Tensor packed = _ctx.NewF32(seq, _dim);
                DirectOps.Zero(_ctx, packed);
                using (Tensor top = packed.Narrow(0, 0, tokens)) Ops.Copy(top, h);
                if (_registerTokens != null)
                    using (Tensor reg = packed.Narrow(0, tokens, _numRegister))
                        Ops.Copy(reg, _registerTokens);
                // The trailing row stays zero.
                h.Dispose();
                h = packed;
            }

            try
            {
                float scale = 1.0f / MathF.Sqrt(_headDim);
                foreach (BlockW bw in _blocks)
                {
                    using (Tensor n1 = DirectOps.RmsNorm(_ctx, h, bw.Norm1, _eps))
                    using (Tensor qkv = bw.Qkv.Forward(n1))       // [seq, 3*dim], per-head interleaved
                    {
                        Tensor q = NormHeadsNoGain(SplitInterleaved(qkv, seq, 0), seq);
                        Tensor k = NormHeadsNoGain(SplitInterleaved(qkv, seq, 1), seq);
                        using Tensor v = SplitInterleaved(qkv, seq, 2);
                        RopeHalf(q, cos, sin, seq, _heads, _headDim, _rotDim);
                        RopeHalf(k, cos, sin, seq, _heads, _headDim, _rotDim);

                        using (Tensor merged = DirectOps.Attention(_ctx, q, k, v, _heads, _headDim, scale))
                        using (Tensor attn = bw.Out.Forward(merged))
                        {
                            if (bw.Scale1 != null) DirectOps.MulColsRows(_ctx, attn, bw.Scale1);
                            Ops.Add(h, h, attn);
                        }
                        q.Dispose();
                        k.Dispose();
                    }

                    using (Tensor n2 = DirectOps.RmsNorm(_ctx, h, bw.Norm2, _eps))
                    using (Tensor gv = bw.W1.Forward(n2))         // [seq, 2*inner]
                    {
                        using Tensor act = ActivateMlp(_ctx, gv, _inner, fuseAccelerated: _ctx.IsMlx);
                        using Tensor ff = bw.W2.Forward(act);
                        if (bw.Scale2 != null) DirectOps.MulColsRows(_ctx, ff, bw.Scale2);
                        Ops.Add(h, h, ff);
                    }
                }

                // norm_out is a LayerNorm (mean-centred, with bias), not an RMSNorm.
                using Tensor normed = DirectOps.LayerNorm(_ctx, h, _normOutW, _normOutB, _eps);
                using Tensor proj = _projOut.Forward(normed);      // [seq, patchDim]
                // Drop the suffix tokens; only the real patches carry pixels.
                using Tensor patches = Rows(proj, 0, tokens);
                return DirectOps.ToArray(patches);
            }
            finally
            {
                h.Dispose();
            }
        }

        private static Tensor Rows(Tensor src, int from, int count)
        {
            using var view = src.Narrow(0, from, count);
            return Ops.NewContiguous(view);
        }

        private static Tensor Cols(Tensor src, int from, int count)
        {
            using var view = src.Narrow(1, from, count);
            return Ops.NewContiguous(view);
        }

        internal void ReleaseDeviceResidency()
        {
            if (!_disposed) _ctx.ReleaseDeviceResidency();
        }

        /// <summary>SwiGLU over a [gate | value] projection: silu(gate) * value. The device split kernel reads [gate | value]
        /// directly and keeps the same SiLU-then-multiply arithmetic (no two full activation copies and no separate multiply
        /// pass over the large FFN width).</summary>
        internal static Tensor ActivateMlp(DirectContext context, Tensor gv, int ffn, bool fuseAccelerated = true)
        {
            if (fuseAccelerated && context.IsAccelerated)
                return Ops.SiLUMulSplit(null, gv, ffn);

            using Tensor gate = Columns(gv, 0, ffn);
            using Tensor val = Columns(gv, ffn, ffn);
            Tensor act = DirectOps.Silu(context, gate);
            try
            {
                Ops.Mul(act, act, val);
                return act;
            }
            catch { act.Dispose(); throw; }
        }

        private static Tensor Columns(Tensor src, int from, int count)
        {
            using var view = src.Narrow(1, from, count);
            return Ops.NewContiguous(view);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            foreach (var l in _owned) l.Dispose();
            _owned.Clear();
            _tensorOps.Dispose();
            _ctx.Dispose();
        }
    }
}
