// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Wan 2.1 DiT (WanModel): N single-stream blocks with self-attention (3D RoPE),
// cross-attention over the UMT5 conditioning, and AdaLN time modulation. One
// Predict call = one denoising-step velocity prediction, executed as ONE native
// resident-weight ggml graph (TSGgml_WanDitForward; persistent + CUDA-graph
// captured per shape on CUDA). Weight matrices are bound straight from the GGUF
// mmap in their stored quantization; small tensors (norm gains, modulation,
// biases) are handed over as stable F32 pointers.
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models.WanVideo
{
    internal sealed class WanDiT : IWanDit, IGgmlHostMemoryOwner
    {
        public const int FreqDim = 256;
        public const int TextDim = 4096;
        public const int HeadDim = 128;
        public const int PatchH = 2;
        public const int PatchW = 2;

        private readonly GgufFile _gguf;

        public int Dim { get; }
        public int Layers { get; }
        public int Heads { get; }
        public int Ff { get; }
        public int InDim { get; }      // latent channels (16 for T2V)
        public int OutDim { get; }

        private readonly Dictionary<string, IntPtr> _f32Copies = new();
        private readonly List<IntPtr> _f32Owned = new();
        private WanDitBlockW[] _blocks;

        public WanDiT(GgufFile ditGguf)
        {
            _gguf = ditGguf;

            int layers = 0;
            foreach (var name in _gguf.Tensors.Keys)
            {
                if (!name.StartsWith("blocks.", StringComparison.Ordinal)) continue;
                int dot = name.IndexOf('.', 7);
                if (dot > 7 && int.TryParse(name.AsSpan(7, dot - 7), out int idx))
                    layers = Math.Max(layers, idx + 1);
            }
            if (layers == 0)
                throw new InvalidDataException("Wan DiT GGUF has no 'blocks.N.*' tensors.");
            Layers = layers;

            var q0 = _gguf.Tensors["blocks.0.self_attn.q.weight"];
            Dim = (int)q0.Shape[0];
            Heads = Dim / HeadDim;
            Ff = (int)_gguf.Tensors["blocks.0.ffn.0.weight"].Shape[1];

            var patch = _gguf.Tensors["patch_embedding.weight"];
            // 5D [kw, kh, kd, ic, oc]; kd is 1 for Wan 2.1, so tokens are ic*kh*kw wide.
            InDim = (int)patch.Shape[3];
            OutDim = (int)(_gguf.Tensors["head.head.weight"].Shape[1] / (PatchH * PatchW));

            if (_gguf.Tensors.ContainsKey("blocks.0.cross_attn.k_img.weight"))
                throw new NotSupportedException(
                    "This Wan DiT is a Wan 2.1 image-to-video checkpoint (CLIP-conditioned k_img/v_img " +
                    "cross-attention), which TensorSharp does not support. For image-to-video use the " +
                    "Wan 2.2 models (Wan2.2-TI2V-5B or Wan2.2-I2V-A14B), which need no image encoder.");

            Console.WriteLine($"  Wan DiT: {Layers} blocks, dim={Dim}, heads={Heads}, ffn={Ff}, in={InDim}");
            _blocks = BuildBlockTable();
        }

        /// <summary>Tokens per patch (in-feature width of the patch-embedding matmul).</summary>
        public int InTok => InDim * PatchH * PatchW;
        public int OutTok => OutDim * PatchH * PatchW;

        private WanDitBlockW[] BuildBlockTable()
        {
            var blocks = new WanDitBlockW[Layers];
            for (int l = 0; l < Layers; l++)
            {
                string p = $"blocks.{l}.";
                blocks[l] = new WanDitBlockW
                {
                    Modulation = F32Ptr($"{p}modulation"),
                    SQ = W($"{p}self_attn.q.weight", $"{p}self_attn.q.bias"),
                    SK = W($"{p}self_attn.k.weight", $"{p}self_attn.k.bias"),
                    SV = W($"{p}self_attn.v.weight", $"{p}self_attn.v.bias"),
                    SO = W($"{p}self_attn.o.weight", $"{p}self_attn.o.bias"),
                    SNormQ = F32Ptr($"{p}self_attn.norm_q.weight"),
                    SNormK = F32Ptr($"{p}self_attn.norm_k.weight"),
                    Norm3W = F32Ptr($"{p}norm3.weight"),
                    Norm3B = F32Ptr($"{p}norm3.bias"),
                    XQ = W($"{p}cross_attn.q.weight", $"{p}cross_attn.q.bias"),
                    XK = W($"{p}cross_attn.k.weight", $"{p}cross_attn.k.bias"),
                    XV = W($"{p}cross_attn.v.weight", $"{p}cross_attn.v.bias"),
                    XO = W($"{p}cross_attn.o.weight", $"{p}cross_attn.o.bias"),
                    XNormQ = F32Ptr($"{p}cross_attn.norm_q.weight"),
                    XNormK = F32Ptr($"{p}cross_attn.norm_k.weight"),
                    Ffn0 = W($"{p}ffn.0.weight", $"{p}ffn.0.bias"),
                    Ffn2 = W($"{p}ffn.2.weight", $"{p}ffn.2.bias"),
                };
            }
            return blocks;
        }

        /// <summary>
        /// One denoising-step forward: patchified latent tokens in, velocity tokens out.
        /// </summary>
        /// <param name="xTokens">[InTok * seq] patchified latent tokens.</param>
        /// <param name="context">[TextDim * 512] zero-padded UMT5 conditioning.</param>
        /// <param name="timestep">timestep in [0, 1000].</param>
        /// <param name="rope">precomputed cos/sin tables for this grid.</param>
        /// <param name="seq">token count.</param>
        /// <param name="seq0">tokens [0, seq0) are modulated at <paramref name="timestep0"/>
        /// instead (Wan 2.2 TI2V i2v: the conditioning frame's tokens run at t=0). 0 = uniform.</param>
        /// <param name="timestep0">timestep for the first <paramref name="seq0"/> tokens.</param>
        /// <returns>[OutTok * seq] velocity tokens.</returns>
        public float[] Predict(float[] xTokens, float[] context, float timestep, WanRope rope, int seq,
                               int seq0 = 0, float timestep0 = 0f)
        {
            var tsin = new float[FreqDim];
            FillSinusoid(tsin, timestep);
            float[] tsin0 = null;
            if (seq0 > 0)
            {
                tsin0 = new float[FreqDim];
                FillSinusoid(tsin0, timestep0);
            }

            var outTokens = new float[(long)OutTok * seq];
            var handles = new List<GCHandle>();
            IntPtr Pin<T>(T[] a) where T : struct
            {
                var h = GCHandle.Alloc(a, GCHandleType.Pinned);
                handles.Add(h);
                return h.AddrOfPinnedObject();
            }
            try
            {
                var d = new WanDitForwardArgs
                {
                    X = Pin(xTokens),
                    Out = Pin(outTokens),
                    Context = Pin(context),
                    TSin = Pin(tsin),
                    TSin0 = tsin0 != null ? Pin(tsin0) : IntPtr.Zero,
                    CosF = Pin(rope.Cos),
                    SinF = Pin(rope.Sin),
                    Patch = PatchW5D(),
                    Text0 = W("text_embedding.0.weight", "text_embedding.0.bias"),
                    Text2 = W("text_embedding.2.weight", "text_embedding.2.bias"),
                    Time0 = W("time_embedding.0.weight", "time_embedding.0.bias"),
                    Time2 = W("time_embedding.2.weight", "time_embedding.2.bias"),
                    TProj = W("time_projection.1.weight", "time_projection.1.bias"),
                    Head = W("head.head.weight", "head.head.bias"),
                    HeadMod = F32Ptr("head.modulation"),
                    Blocks = Pin(_blocks),
                    NumLayers = Layers,
                    Dim = Dim,
                    Ff = Ff,
                    Heads = Heads,
                    HeadDim = HeadDim,
                    Seq = seq,
                    CtxLen = WanTextEncoder.TextLen,
                    FreqDim = FreqDim,
                    TextDim = TextDim,
                    Seq0 = tsin0 != null ? seq0 : 0,
                    Eps = 1e-6f,
                    StructBytes = Marshal.SizeOf<WanDitForwardArgs>(),
                };
                if (!GgmlBasicOps.TryWanDitForward(in d))
                    throw new InvalidOperationException("Wan DiT forward failed (see native error above).");
            }
            finally
            {
                foreach (var h in handles) h.Free();
            }

            return ReorderHeadOutput(outTokens, seq, OutDim);
        }

        // The head's output features are laid out (p_t, p_h, p_w, c) with the CHANNEL
        // fastest (Wan unpatchify: view(*grid, *patch_size, c)), while the input patch
        // features are (c, kh, kw) with kw fastest (the conv kernel layout). Reorder the
        // velocity into the input layout so token-space Euler stepping and Unpatchify
        // use one convention. Shared with the direct-backend WanDirectDiT.
        internal static float[] ReorderHeadOutput(float[] outTokens, int seq, int oc)
        {
            int outTok = oc * PatchH * PatchW;
            var reordered = new float[outTokens.Length];
            System.Threading.Tasks.Parallel.For(0, seq, tok =>
            {
                long src = (long)tok * outTok, dst = src;
                for (int kh = 0; kh < PatchH; kh++)
                    for (int kw = 0; kw < PatchW; kw++)
                        for (int c = 0; c < oc; c++)
                            reordered[dst + c * PatchH * PatchW + kh * PatchW + kw] =
                                outTokens[src + (kh * PatchW + kw) * oc + c];
            });
            return reordered;
        }

        // Wan sinusoidal_embedding_1d(dim=256, t): [cos(t*w_j) | sin(t*w_j)],
        // w_j = 10000^(-j/half).
        internal static void FillSinusoid(float[] dst, float t)
        {
            int half = dst.Length / 2;
            for (int j = 0; j < half; j++)
            {
                float w = MathF.Pow(10000f, -(float)j / half);
                dst[j] = MathF.Cos(t * w);
                dst[half + j] = MathF.Sin(t * w);
            }
        }

        // The 5D patch-embedding conv [kw=2, kh=2, kd=1, ic, oc] is, in memory,
        // exactly the [ic*kh*kw, oc] matmul weight over tokens whose in-feature
        // index is kw + 2*kh + 4*ic (see WanVideoPipeline.Patchify).
        private WanW PatchW5D()
        {
            var info = _gguf.Tensors["patch_embedding.weight"];
            long inTok = 1;
            for (int i = 0; i + 1 < info.Shape.Length; i++) inTok *= (long)info.Shape[i];
            long oc = (long)info.Shape[info.Shape.Length - 1];
            long blockSize = GgufFile.GetBlockSize(info.Type);
            if (blockSize > 1 && inTok % blockSize != 0)
                throw new NotSupportedException(
                    $"patch_embedding.weight is {info.Type} with block size {blockSize}, which does not divide " +
                    $"the {inTok}-wide patch rows; requantize the DiT with an F16/F32 patch embedding.");
            _gguf.TryGetTensorDataPointer(info, out IntPtr p);
            var bias = _gguf.Tensors.ContainsKey("patch_embedding.bias")
                ? F32Ptr("patch_embedding.bias") : IntPtr.Zero;
            return new WanW
            {
                W = p,
                Type = (int)info.Type,
                Ne0 = inTok,
                Ne1 = oc,
                Bytes = _gguf.GetTensorByteCount(info),
                B = bias,
            };
        }

        private WanW W(string weightName, string biasName)
        {
            var info = _gguf.Tensors[weightName];
            _gguf.TryGetTensorDataPointer(info, out IntPtr p);
            long ne0 = (long)info.Shape[0];
            long ne1 = info.Shape.Length > 1 ? (long)info.Shape[1] : 1;
            IntPtr bias = biasName != null && _gguf.Tensors.ContainsKey(biasName)
                ? F32Ptr(biasName) : IntPtr.Zero;
            return new WanW
            {
                W = p,
                Type = (int)info.Type,
                Ne0 = ne0,
                Ne1 = ne1,
                Bytes = _gguf.GetTensorByteCount(info),
                B = bias,
            };
        }

        // Stable F32 pointer: zero-copy mmap for F32 tensors, one-time dequantized
        // unmanaged copy otherwise (the native side resident-caches by pointer).
        // Non-F32 tensors (the Wan 2.2 GGUFs store biases/norms/modulation as F16)
        // go through NativeDequant — ReadTensorDataToFloat32Native is a raw byte
        // copy that only ever fits already-F32 data.
        private IntPtr F32Ptr(string name)
        {
            lock (_f32Copies)
            {
                if (_f32Copies.TryGetValue(name, out IntPtr cached)) return cached;
                var info = _gguf.Tensors[name];
                IntPtr result;
                if (info.Type == GgmlTensorType.F32)
                {
                    _gguf.TryGetTensorDataPointer(info, out result);
                }
                else
                {
                    long nel = info.NumElements;
                    long srcBytes = _gguf.GetTensorByteCount(info);
                    result = Marshal.AllocHGlobal((IntPtr)(nel * sizeof(float)));
                    _f32Owned.Add(result);
                    IntPtr tmp = Marshal.AllocHGlobal((IntPtr)srcBytes);
                    try
                    {
                        _gguf.ReadTensorDataToNative(info, tmp, srcBytes);
                        NativeDequant.DequantizeToFloat32Native((int)info.Type, tmp, result, nel);
                    }
                    finally { Marshal.FreeHGlobal(tmp); }
                }
                _f32Copies[name] = result;
                return result;
            }
        }

        /// <summary>The F32 copies this transformer made (the GGUF mapping is the model's).</summary>
        void IGgmlHostMemoryOwner.CollectHostMemory(List<(IntPtr Base, long Length)> ranges, List<IntPtr> keys) => keys.AddRange(_f32Owned);

        public void Dispose()
        {
            foreach (var p in _f32Owned)
            {
                GgmlBasicOps.InvalidateHostBuffer(p);   // see WanVae.Dispose
                Marshal.FreeHGlobal(p);
            }
            _f32Owned.Clear();
            _f32Copies.Clear();
            // _gguf is owned by WanVideoModel (it is the model's base GGUF).
        }
    }

    /// <summary>
    /// Precomputed 3D RoPE tables for one (t_len, h_len, w_len) grid, in the
    /// pair-duplicated [head_dim, seq] layout the native kernel consumes
    /// (cos[2i] == cos[2i+1] == cos_i). Axis split 44/42/42 over head_dim 128:
    /// pair p &lt; 22 rotates by the frame index, p in [22,43) by the row,
    /// p in [43,64) by the column (theta 10000, per-axis exponent 2j/d_axis).
    /// </summary>
    internal sealed class WanRope
    {
        public float[] Cos;   // [head_dim * seq]
        public float[] Sin;

        public static WanRope Build(int tLen, int hLen, int wLen, int headDim = WanDiT.HeadDim)
        {
            int dT = headDim - 2 * (headDim / 3 - headDim / 3 % 2);  // 44 for 128
            int dH = (headDim - dT) / 2;                             // 42
            int dW = dH;
            int half = headDim / 2;
            int pT = dT / 2, pH = dH / 2, pW = dW / 2;

            long seq = (long)tLen * hLen * wLen;
            var cos = new float[seq * headDim];
            var sin = new float[seq * headDim];
            long tok = 0;
            for (int ti = 0; ti < tLen; ti++)
            {
                for (int hi = 0; hi < hLen; hi++)
                {
                    for (int wi = 0; wi < wLen; wi++)
                    {
                        long o = tok * headDim;
                        int p = 0;
                        WriteAxis(cos, sin, o, ref p, dT, ti);
                        WriteAxis(cos, sin, o, ref p, dH, hi);
                        WriteAxis(cos, sin, o, ref p, dW, wi);
                        tok++;
                    }
                }
            }
            return new WanRope { Cos = cos, Sin = sin };
        }

        private static void WriteAxis(float[] cos, float[] sin, long baseOff, ref int pair, int dAxis, int pos)
        {
            int n = dAxis / 2;
            for (int j = 0; j < n; j++)
            {
                double freq = Math.Pow(10000.0, -2.0 * j / dAxis);
                double ang = pos * freq;
                float c = (float)Math.Cos(ang), s = (float)Math.Sin(ang);
                cos[baseOff + 2 * pair] = c; cos[baseOff + 2 * pair + 1] = c;
                sin[baseOff + 2 * pair] = s; sin[baseOff + 2 * pair + 1] = s;
                pair++;
            }
        }
    }
}
