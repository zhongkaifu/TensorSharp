// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ============================================================================
// Fused whole-VAE graph for Qwen-Image (TSGgml_QwenVaeRun).
//
// The legacy path runs the VAE as a C# op chain where only the convs go to the
// device — each conv re-uploads its weights and the full feature map and
// downloads the result, and SiLU / channel-RMSNorm / nearest-upsample run as C#
// loops over the host arrays in between. At ~1 MP that is GBs of PCIe traffic
// plus CPU elementwise passes (~21 s encode + ~27 s decode at 928x704).
//
// This class emits the SAME topology as VaeReferenceMath (the verified
// bit-exact-vs-diffusers reference) as a flat op list that the native kernel
// executes as ONE device-resident ggml graph: features never leave the GPU,
// weights are bound resident from stable unmanaged buffers (uploaded once,
// reused across encode/decode/edits), and convs use ggml_conv_2d_direct (no
// materialized im2col, so no band-tiling). Per call: one input upload, one
// compute, one sync, one output download.
//
// The op lists are resolution-independent (shapes flow from the input tensor),
// so one build serves every image size. Correctness gate: `vae-verify` runs the
// fused path against the diffusers oracle npy files.
// ============================================================================
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using TensorSharp.GGML;

namespace TensorSharp.Models.QwenImage
{
    internal sealed class QwenImageVaeGraph : IDisposable
    {
        private const int KConv = 0, KNorm = 1, KSilu = 2, KUp = 3, KSave = 4, KAdd = 5, KAttn = 6;
        private const int ZDim = 16;

        private readonly List<IntPtr> _allocs = new();
        private QwenVaeWeightRef[] _weights;
        private QwenVaeOp[] _encodeOps, _decodeOps;
        private bool _disposed;

        public static QwenImageVaeGraph TryBuild(VaeWeights w)
        {
            var g = new QwenImageVaeGraph();
            try
            {
                g.Build(w);
                return g;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  [vae-fused] build failed, using per-conv path: {ex.Message}");
                g.Dispose();
                return null;
            }
        }

        public unsafe bool TryEncode(float[] chw, int H, int W, out float[] z32, out int lh, out int lw)
        {
            lh = H / 8; lw = W / 8;
            z32 = null;
            if (H % 8 != 0 || W % 8 != 0) return false;
            var outp = new float[2L * ZDim * lh * lw];
            if (!Run(_encodeOps, chw, W, H, 3, outp)) return false;
            z32 = outp;
            return true;
        }

        public unsafe bool TryDecode(float[] latent, int lh, int lw, out float[] rgb, out int H, out int W)
        {
            H = lh * 8; W = lw * 8;
            rgb = null;
            var outp = new float[3L * H * W];
            if (!Run(_decodeOps, latent, lw, lh, ZDim, outp)) return false;
            rgb = outp;
            return true;
        }

        private unsafe bool Run(QwenVaeOp[] ops, float[] input, int inW, int inH, int inC, float[] output)
        {
            fixed (float* ip = input, op = output)
            fixed (QwenVaeOp* opsPtr = ops)
            fixed (QwenVaeWeightRef* wPtr = _weights)
            {
                var a = new QwenVaeArgs
                {
                    Input = (IntPtr)ip, InW = inW, InH = inH, InC = inC,
                    Output = (IntPtr)op, OutLen = output.LongLength,
                    Ops = (IntPtr)opsPtr, NumOps = ops.Length,
                    Weights = (IntPtr)wPtr, NumWeights = _weights.Length,
                    StructBytes = Marshal.SizeOf<QwenVaeArgs>(),
                };
                return GgmlBasicOps.TryQwenVaeRun(in a);
            }
        }

        // ---- build ------------------------------------------------------------

        private void Build(VaeWeights w)
        {
            var weights = new List<QwenVaeWeightRef>();

            int Reg(float[] data)
            {
                long bytes = data.LongLength * sizeof(float);
                IntPtr p = Marshal.AllocHGlobal((IntPtr)bytes);
                _allocs.Add(p);
                Marshal.Copy(data, 0, p, data.Length);
                weights.Add(new QwenVaeWeightRef { Data = p, Bytes = bytes });
                return weights.Count - 1;
            }
            // Causal Conv3d on T=1 uses only the last temporal slice (kd = KD-1) of the
            // 5D (oc,ic,kd,kh,kw) weight — same slicing as VaeReferenceMath.CausalConv3dT1.
            int RegSlice(string name, int OC, int IC, int KD, int KH, int KW)
            {
                float[] w5d = w.Get(name);
                if (KD == 1) return Reg(w5d);
                var slice = new float[(long)OC * IC * KH * KW];
                int khw = KH * KW, kdhw = KD * KH * KW;
                Parallel.For(0, OC, oc =>
                {
                    for (int ic = 0; ic < IC; ic++)
                    {
                        long src = ((long)oc * IC + ic) * kdhw + (long)(KD - 1) * khw;
                        long dst = ((long)oc * IC + ic) * khw;
                        Array.Copy(w5d, src, slice, dst, khw);
                    }
                });
                return Reg(slice);
            }

            List<QwenVaeOp> ops = null;
            void Emit(int kind, int wi = -1, int bi = -1, int oc = 0, int ic = 0, int kh = 0, int kw = 0,
                      int sh = 1, int sw = 1, int pt = 0, int pb = 0, int pl = 0, int pr = 0,
                      int src = 0, int dst = 0, int aux = 0) =>
                ops.Add(new QwenVaeOp { Kind = kind, W = wi, B = bi, Oc = oc, Ic = ic, Kh = kh, Kw = kw,
                                        Sh = sh, Sw = sw, Pt = pt, Pb = pb, Pl = pl, Pr = pr,
                                        Src = src, Dst = dst, Aux = aux });

            // conv from a causal-3D weight (sliced) with symmetric spatial pad
            void CausalConv(string prefix, int oc, int ic, int kd, int k, int pad, int src = 0, int dst = 0) =>
                Emit(KConv, RegSlice(prefix + ".weight", oc, ic, kd, k, k), Reg(w.Get(prefix + ".bias")),
                     oc, ic, k, k, pt: pad, pb: pad, pl: pad, pr: pad, src: src, dst: dst);
            void Norm(string gammaName) => Emit(KNorm, Reg(w.Get(gammaName)));
            void Silu() => Emit(KSilu);

            void ResidualBlock(string prefix, int inDim, int outDim)
            {
                if (inDim != outDim) CausalConv(prefix + ".shortcut", outDim, inDim, 1, 1, 0, src: 0, dst: 1);
                else Emit(KSave, src: 0, dst: 1);
                Norm(prefix + ".residual.0.gamma");
                Silu();
                CausalConv(prefix + ".residual.2", outDim, inDim, 3, 3, 1);
                Norm(prefix + ".residual.3.gamma");
                Silu();
                CausalConv(prefix + ".residual.6", outDim, outDim, 3, 3, 1);
                Emit(KAdd, src: 0, dst: 0, aux: 1);
            }
            void AttentionBlock(string prefix, int c)
            {
                Emit(KSave, src: 0, dst: 1);
                Norm(prefix + ".norm.gamma");
                Emit(KConv, Reg(w.Get(prefix + ".to_qkv.weight")), Reg(w.Get(prefix + ".to_qkv.bias")), 3 * c, c, 1, 1);
                Emit(KAttn, oc: c);
                Emit(KConv, Reg(w.Get(prefix + ".proj.weight")), Reg(w.Get(prefix + ".proj.bias")), c, c, 1, 1);
                Emit(KAdd, src: 0, dst: 0, aux: 1);
            }
            void MidBlock(string prefix, int dim)
            {
                ResidualBlock(prefix + ".0", dim, dim);
                AttentionBlock(prefix + ".1", dim);
                ResidualBlock(prefix + ".2", dim, dim);
            }

            // ---- encoder (mirrors VaeReferenceMath.Encode after pixel normalize) ----
            ops = new List<QwenVaeOp>();
            CausalConv("encoder.conv1", 96, 3, 3, 3, 1);
            int[] eDims = { 96, 96, 192, 384, 384 };
            int idx = 0;
            for (int i = 0; i < 4; i++)
            {
                ResidualBlock($"encoder.downsamples.{idx++}", eDims[i], eDims[i + 1]);
                ResidualBlock($"encoder.downsamples.{idx++}", eDims[i + 1], eDims[i + 1]);
                if (i != 3)
                {
                    // Downsample: ZeroPad2d((0,1,0,1)) + conv stride 2, pad 0
                    string p = $"encoder.downsamples.{idx++}";
                    Emit(KConv, Reg(w.Get(p + ".resample.1.weight")), Reg(w.Get(p + ".resample.1.bias")),
                         eDims[i + 1], eDims[i + 1], 3, 3, sh: 2, sw: 2, pt: 0, pb: 1, pl: 0, pr: 1);
                }
            }
            MidBlock("encoder.middle", 384);
            Norm("encoder.head.0.gamma");
            Silu();
            CausalConv("encoder.head.2", 2 * ZDim, 384, 3, 3, 1);
            CausalConv("conv1", 2 * ZDim, 2 * ZDim, 1, 1, 0);   // quant_conv
            _encodeOps = ops.ToArray();

            // ---- decoder (mirrors VaeReferenceMath.Decode before the [-1,1]->[0,1] map) ----
            ops = new List<QwenVaeOp>();
            CausalConv("conv2", ZDim, ZDim, 1, 1, 0);           // post_quant_conv
            CausalConv("decoder.conv1", 384, ZDim, 3, 3, 1);
            MidBlock("decoder.middle", 384);
            int[] inDims = { 384, 192, 192, 96 };
            int[] outDims = { 384, 384, 192, 96 };
            idx = 0;
            for (int i = 0; i < 4; i++)
            {
                int cur = inDims[i];
                for (int r = 0; r < 3; r++)   // num_res_blocks + 1
                {
                    ResidualBlock($"decoder.upsamples.{idx++}", cur, outDims[i]);
                    cur = outDims[i];
                }
                if (i != 3)
                {
                    // Upsample: nearest x2 + conv(dim -> dim/2), pad 1
                    string p = $"decoder.upsamples.{idx++}";
                    Emit(KUp);
                    Emit(KConv, Reg(w.Get(p + ".resample.1.weight")), Reg(w.Get(p + ".resample.1.bias")),
                         outDims[i] / 2, outDims[i], 3, 3, pt: 1, pb: 1, pl: 1, pr: 1);
                }
            }
            Norm("decoder.head.0.gamma");
            Silu();
            CausalConv("decoder.head.2", 3, 96, 3, 3, 1);
            _decodeOps = ops.ToArray();

            _weights = weights.ToArray();
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            foreach (var p in _allocs) Marshal.FreeHGlobal(p);
            _allocs.Clear();
            _weights = null;
        }
    }
}
